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
    Serial Eigen backend for the coupled p-U system.

    The Eigen headers are included first so that none of the OpenFOAM
    macros can interfere with them.

\*---------------------------------------------------------------------------*/

#include <vector>
#include <Eigen/Sparse>
#include <Eigen/SparseLU>
#include <Eigen/IterativeLinearSolvers>

#include "eigenCoupledSolver.H"
#include "IOstreams.H"
#include "error.H"

typedef Eigen::SparseMatrix<double, Eigen::ColMajor, int> coupledSpMat;
typedef Eigen::SparseLU<coupledSpMat, Eigen::COLAMDOrdering<int>> coupledLU;
typedef Eigen::Triplet<double, int> coupledTriplet;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::eigenCoupledSolver::eigenCoupledSolver
(
    const dictionary& dict,
    const word& backend
)
:
    name_(backend == "BiCGSTAB" ? "EigenBiCGSTAB" : "EigenSparseLU"),
    tolerance_(dict.lookupOrDefault<scalar>("tolerance", 1e-10)),
    maxIter_(dict.lookupOrDefault<label>("maxLinearIter", 2000)),
    dropTolerance_(dict.lookupOrDefault<scalar>("dropTolerance", 1e-5)),
    fillFactor_(dict.lookupOrDefault<label>("fillFactor", 20)),
    patternAnalysed_(false),
    luPtr_(nullptr)
{
    if (backend != "SparseLU" && backend != "BiCGSTAB")
    {
        FatalErrorInFunction
            << "Unknown Eigen backend " << backend << nl
            << "Valid options are SparseLU and BiCGSTAB"
            << exit(FatalError);
    }

    if (name_ == "EigenSparseLU")
    {
        luPtr_ = static_cast<void*>(new coupledLU());
    }
}


Foam::eigenCoupledSolver::~eigenCoupledSolver()
{
    if (luPtr_)
    {
        delete static_cast<coupledLU*>(luPtr_);
        luPtr_ = nullptr;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::label Foam::eigenCoupledSolver::solve
(
    const coupledLinearSystem& sys,
    scalarField& x
)
{
    const int n = static_cast<int>(sys.nGlobalRows());

    if (sys.nLocalRows() != sys.nGlobalRows())
    {
        FatalErrorInFunction
            << "The Eigen backend is serial only" << exit(FatalError);
    }

    std::vector<coupledTriplet> trips;
    trips.reserve(static_cast<std::size_t>(sys.nEntries()));

    for (label k = 0; k < sys.nEntries(); k++)
    {
        trips.push_back
        (
            coupledTriplet
            (
                static_cast<int>(sys.row(k)),
                static_cast<int>(sys.col(k)),
                sys.val(k)
            )
        );
    }

    coupledSpMat A(n, n);
    A.setFromTriplets(trips.begin(), trips.end());
    A.makeCompressed();

    Eigen::VectorXd b(n);
    for (int i = 0; i < n; i++)
    {
        b(i) = sys.rhs()[i];
    }

    Eigen::VectorXd sol(n);

    label nIter = 1;

    if (name_ == "EigenSparseLU")
    {
        coupledLU& lu = *static_cast<coupledLU*>(luPtr_);

        if (!patternAnalysed_)
        {
            // NB: SparseLU only marks itself initialised after factorize(),
            // so info() must not be queried here
            lu.analyzePattern(A);
            patternAnalysed_ = true;
        }

        lu.factorize(A);

        if (lu.info() != Eigen::Success)
        {
            FatalErrorInFunction
                << "Eigen SparseLU factorisation failed (singular matrix?): "
                << lu.lastErrorMessage().c_str() << nl
                << "If the case has no fixedValue pressure patch, check that"
                << " the pressure reference cell is being pinned."
                << exit(FatalError);
        }

        sol = lu.solve(b);

        if (lu.info() != Eigen::Success)
        {
            FatalErrorInFunction
                << "Eigen SparseLU back-substitution failed"
                << exit(FatalError);
        }
    }
    else
    {
        Eigen::BiCGSTAB<coupledSpMat, Eigen::IncompleteLUT<double>> bicg;
        bicg.preconditioner().setDroptol(dropTolerance_);
        bicg.preconditioner().setFillfactor(static_cast<int>(fillFactor_));
        bicg.setTolerance(tolerance_);
        bicg.setMaxIterations(static_cast<int>(maxIter_));
        bicg.compute(A);

        if (bicg.info() != Eigen::Success)
        {
            FatalErrorInFunction
                << "Eigen IncompleteLUT preconditioner setup failed"
                << exit(FatalError);
        }

        Eigen::VectorXd x0(n);
        for (int i = 0; i < n; i++)
        {
            x0(i) = x[i];
        }

        sol = bicg.solveWithGuess(b, x0);

        nIter = static_cast<label>(bicg.iterations());

        if (bicg.info() != Eigen::Success)
        {
            WarningInFunction
                << "Eigen BiCGSTAB did not converge in " << nIter
                << " iterations, residual " << bicg.error() << endl;
        }
    }

    for (int i = 0; i < n; i++)
    {
        x[i] = sol(i);
    }

    return nIter;
}


// ************************************************************************* //
