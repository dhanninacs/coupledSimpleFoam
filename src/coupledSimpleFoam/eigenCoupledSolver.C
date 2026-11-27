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
// Row-major for BiCGSTAB: Eigen multi-threads sparse matrix-vector products
// (OpenMP) for row-major matrices only.
typedef Eigen::SparseMatrix<double, Eigen::RowMajor, int> coupledRowMat;
typedef Eigen::IncompleteLUT<double, int> coupledILUT;

namespace
{
    // A preconditioner for Eigen::BiCGSTAB that applies an IncompleteLUT
    // built elsewhere and kept between solves: Eigen's own compute() would
    // rebuild it for every matrix, which cost most of each outer iteration.
    class keptILUT
    {
    public:
        typedef double Scalar;
        typedef double RealScalar;
        typedef int StorageIndex;
        enum { ColsAtCompileTime = Eigen::Dynamic, MaxColsAtCompileTime = Eigen::Dynamic };

        keptILUT() : ilut_(nullptr) {}
        template<typename M> keptILUT(const M&) : ilut_(nullptr) {}
        void set(const coupledILUT* ilut) { ilut_ = ilut; }
        template<typename M> keptILUT& analyzePattern(const M&) { return *this; }
        template<typename M> keptILUT& factorize(const M&) { return *this; }
        template<typename M> keptILUT& compute(const M&) { return *this; }
        template<typename Rhs> Eigen::VectorXd solve(const Rhs& b) const
        {
            Eigen::VectorXd x = ilut_->solve(b);
            return x;
        }
        Eigen::ComputationInfo info() const { return Eigen::Success; }

    private:
        const coupledILUT* ilut_;
    };
}

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
    dropTolerance_(dict.lookupOrDefault<scalar>("dropTolerance", 1e-4)),
    fillFactor_(dict.lookupOrDefault<label>("fillFactor", 5)),
    patternAnalysed_(false),
    luPtr_(nullptr),
    ilutPtr_(nullptr),
    pcReuseInterval_(dict.lookupOrDefault<label>("pcReuseInterval", 1)),
    solvesSinceBuild_(0),
    relTol_(-1)
{
    if (pcReuseInterval_ < 1)
    {
        FatalErrorInFunction
            << "pcReuseInterval must be >= 1, not " << pcReuseInterval_
            << exit(FatalError);
    }
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
    if (ilutPtr_)
    {
        delete static_cast<coupledILUT*>(ilutPtr_);
        ilutPtr_ = nullptr;
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
        coupledRowMat Ar(A);
        Ar.makeCompressed();

        // Rebuild the IncompleteLUT on the first solve and every
        // pcReuseInterval solves; in between the kept one preconditions
        // the new matrix.
        if (!ilutPtr_ || solvesSinceBuild_ >= pcReuseInterval_)
        {
            coupledILUT* ilut = static_cast<coupledILUT*>(ilutPtr_);
            if (!ilut)
            {
                ilut = new coupledILUT();
                ilutPtr_ = static_cast<void*>(ilut);
            }
            ilut->setDroptol(dropTolerance_);
            ilut->setFillfactor(static_cast<int>(fillFactor_));
            ilut->compute(Ar);
            if (ilut->info() != Eigen::Success)
            {
                FatalErrorInFunction
                    << "Eigen IncompleteLUT preconditioner setup failed"
                    << exit(FatalError);
            }
            solvesSinceBuild_ = 0;
        }
        solvesSinceBuild_++;

        Eigen::VectorXd x0(n);
        for (int i = 0; i < n; i++)
        {
            x0(i) = x[i];
        }

        Eigen::BiCGSTAB<coupledRowMat, keptILUT> bicg;
        bicg.preconditioner().set(static_cast<const coupledILUT*>(ilutPtr_));
        // Eigen stops at ||r|| <= tol*||b||. The adaptive (Eisenstat-Walker)
        // tolerance is relative to this solve's initial residual
        // ||b - A x0||, as on the PETSc path: measured against ||b|| a good
        // initial guess would make a loose tolerance accept no update at all
        // and the outer loop would plateau.
        scalar tol = tolerance_;
        if (relTol_ > 0)
        {
            const scalar bNorm = b.norm();
            const scalar r0Norm = (b - Ar*x0).norm();
            tol = bNorm > 0 ? relTol_*r0Norm/bNorm : relTol_;
            tol = max(tol, scalar(1e-14));
        }
        bicg.setTolerance(tol);
        bicg.setMaxIterations(static_cast<int>(maxIter_));
        bicg.compute(Ar);

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
