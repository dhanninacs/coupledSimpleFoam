/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | coupledSimpleFoam
   \\    /   O peration     | Fully coupled steady incompressible p-U solver
    \\  /    A nd           | for OpenFOAM
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Copyright
    Copyright (C) 2026 dhanninacs

License
    This file is part of coupledSimpleFoam, a solver application for OpenFOAM.

    coupledSimpleFoam is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    coupledSimpleFoam is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License along
    with coupledSimpleFoam.  If not, see <http://www.gnu.org/licenses/>.
    A verbatim copy is distributed in the LICENSE file at the root of this
    repository.

Description
    Implementation of Foam::coupledLinearSystem and of the linear-algebra
    backend factory Foam::coupledSolver::New.

\*---------------------------------------------------------------------------*/

#include "coupledLinearSystem.H"
#include "eigenCoupledSolver.H"
#include "petscCoupledSolver.H"
#include "amgxCoupledSolver.H"
#include "Pstream.H"
#include "error.H"
#include "SubList.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::coupledLinearSystem::coupledLinearSystem
(
    const label blockSize,
    const label rowStart,
    const label nLocalRows,
    const label nGlobalRows
)
:
    blockSize_(blockSize),
    rowStart_(rowStart),
    nLocalRows_(nLocalRows),
    nGlobalRows_(nGlobalRows),
    rows_(),
    cols_(),
    vals_(),
    rhs_(nLocalRows, 0.0),
    rowOrder_(),
    rowFirst_(),
    groupedEntries_(-1),
    colsByRow_(),
    valsByRow_(),
    valsStale_(true)
{}


void Foam::coupledLinearSystem::groupByRow() const
{
    if (groupedEntries_ == rows_.size())
    {
        return;
    }
    // Counting sort of the triplet indices by local row.
    rowFirst_.setSize(nLocalRows_ + 1);
    rowFirst_ = 0;
    forAll(rows_, k)
    {
        rowFirst_[rows_[k] - rowStart_ + 1]++;
    }
    for (label i = 0; i < nLocalRows_; i++)
    {
        rowFirst_[i + 1] += rowFirst_[i];
    }
    rowOrder_.setSize(rows_.size());
    List<label> fill(SubList<label>(rowFirst_, nLocalRows_));
    forAll(rows_, k)
    {
        rowOrder_[fill[rows_[k] - rowStart_]++] = k;
    }
    groupedEntries_ = rows_.size();
    colsByRow_.setSize(rows_.size());
    forAll(rowOrder_, q)
    {
        colsByRow_[q] = cols_[rowOrder_[q]];
    }
    valsStale_ = true;
}


void Foam::coupledLinearSystem::gatherValues() const
{
    groupByRow();
    if (!valsStale_)
    {
        return;
    }
    valsByRow_.setSize(vals_.size());
    const label n = vals_.size();
    #pragma omp parallel for schedule(static)
    for (label q = 0; q < n; q++)
    {
        valsByRow_[q] = vals_[rowOrder_[q]];
    }
    valsStale_ = false;
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::coupledLinearSystem::clear()
{
    rows_.clear();
    cols_.clear();
    vals_.clear();
    rhs_ = 0.0;
    valsStale_ = true;
}


void Foam::coupledLinearSystem::reserve(const label n)
{
    rows_.reserve(n);
    cols_.reserve(n);
    vals_.reserve(n);
}


void Foam::coupledLinearSystem::setRowIdentity
(
    const label gRow,
    const scalar value
)
{
    if (!ownRow(gRow))
    {
        return;
    }

    if (groupedEntries_ == rows_.size() + 1)
    {
        // The pattern of the previous outer iteration, which ended with the
        // identity entry this call is about to append: only this row's
        // triplets, not a scan of all of them, skipping that last index.
        const label i = gRow - rowStart_;
        for (label q = rowFirst_[i]; q < rowFirst_[i + 1]; q++)
        {
            const label k = rowOrder_[q];
            if (k < rows_.size())
            {
                vals_[k] = 0.0;
            }
        }
    }
    else
    {
        forAll(rows_, k)
        {
            if (rows_[k] == gRow)
            {
                vals_[k] = 0.0;
            }
        }
    }

    add(gRow, gRow, 1.0);
    rhs_[gRow - rowStart_] = value;
    valsStale_ = true;
}


void Foam::coupledLinearSystem::multiply
(
    const scalarField& xLocal,
    const Map<scalar>& xRemote,
    scalarField& result
) const
{
    result.setSize(nLocalRows_);
    groupByRow();

    // Off-rank columns must all have values; checked here, outside the
    // parallel loop, where a fatal error cannot be raised.
    forAll(cols_, k)
    {
        const label c = cols_[k];
        if (!ownRow(c) && !xRemote.found(c))
        {
            FatalErrorInFunction
                << "No value available for off-rank column " << c
                << exit(FatalError);
        }
    }

    gatherValues();
    #pragma omp parallel for schedule(static)
    for (label i = 0; i < nLocalRows_; i++)
    {
        scalar s = 0.0;
        for (label q = rowFirst_[i]; q < rowFirst_[i + 1]; q++)
        {
            const label c = colsByRow_[q];
            const scalar xv =
                ownRow(c) ? xLocal[c - rowStart_] : xRemote.find(c)();
            s += valsByRow_[q]*xv;
        }
        result[i] = s;
    }
}


void Foam::coupledLinearSystem::multiplyUniform
(
    const FixedList<scalar, 4>& xRef,
    scalarField& result
) const
{
    result.setSize(nLocalRows_);
    groupByRow();

    gatherValues();
    #pragma omp parallel for schedule(static)
    for (label i = 0; i < nLocalRows_; i++)
    {
        scalar s = 0.0;
        for (label q = rowFirst_[i]; q < rowFirst_[i + 1]; q++)
        {
            s += valsByRow_[q]*xRef[colsByRow_[q] % blockSize_];
        }
        result[i] = s;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::autoPtr<Foam::coupledSolver> Foam::coupledSolver::New
(
    const dictionary& dict
)
{
    const word defaultBackend
    (
        Pstream::parRun() ? word("PETSc") : word("SparseLU")
    );

    const word backend
    (
        dict.lookupOrDefault<word>("linearSolver", defaultBackend)
    );

    if (backend == "PETSc")
    {
        #ifdef HAVE_PETSC
        return autoPtr<coupledSolver>(new petscCoupledSolver(dict));
        #else
        FatalErrorInFunction
            << "linearSolver PETSc requested but coupledSimpleFoam was built"
            << " without PETSc support." << nl
            << "Install petsc-dev (pkg-config PETSc) and rebuild, or select"
            << " a serial Eigen backend." << exit(FatalError);
        return autoPtr<coupledSolver>(nullptr);
        #endif
    }

    if (backend == "AmgX")
    {
        #ifdef HAVE_AMGX
        return autoPtr<coupledSolver>(new amgxCoupledSolver(dict));
        #else
        FatalErrorInFunction
            << "linearSolver AmgX requested but coupledSimpleFoam was built"
            << " without AmgX support." << nl
            << "Build NVIDIA AmgX, set AMGX_DIR to its install prefix and"
            << " rebuild." << exit(FatalError);
        return autoPtr<coupledSolver>(nullptr);
        #endif
    }

    if (Pstream::parRun())
    {
        FatalErrorInFunction
            << "linearSolver " << backend << " is a serial (Eigen) backend"
            << " but this is a parallel run.  Use linearSolver PETSc."
            << exit(FatalError);
    }

    return autoPtr<coupledSolver>(new eigenCoupledSolver(dict, backend));
}


// ************************************************************************* //
