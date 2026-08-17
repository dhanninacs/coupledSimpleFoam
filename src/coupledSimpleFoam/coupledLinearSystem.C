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
#include "Pstream.H"
#include "error.H"

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
    rhs_(nLocalRows, 0.0)
{}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::coupledLinearSystem::clear()
{
    rows_.clear();
    cols_.clear();
    vals_.clear();
    rhs_ = 0.0;
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

    forAll(rows_, k)
    {
        if (rows_[k] == gRow)
        {
            vals_[k] = 0.0;
        }
    }

    add(gRow, gRow, 1.0);
    rhs_[gRow - rowStart_] = value;
}


void Foam::coupledLinearSystem::multiply
(
    const scalarField& xLocal,
    const Map<scalar>& xRemote,
    scalarField& result
) const
{
    result.setSize(nLocalRows_);
    result = 0.0;

    forAll(rows_, k)
    {
        const label c = cols_[k];

        scalar xv;

        if (ownRow(c))
        {
            xv = xLocal[c - rowStart_];
        }
        else
        {
            if (!xRemote.found(c))
            {
                FatalErrorInFunction
                    << "No value available for off-rank column " << c
                    << exit(FatalError);
            }

            xv = xRemote[c];
        }

        result[rows_[k] - rowStart_] += vals_[k]*xv;
    }
}


void Foam::coupledLinearSystem::multiplyUniform
(
    const FixedList<scalar, 4>& xRef,
    scalarField& result
) const
{
    result.setSize(nLocalRows_);
    result = 0.0;

    forAll(rows_, k)
    {
        result[rows_[k] - rowStart_] +=
            vals_[k]*xRef[cols_[k] % blockSize_];
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
