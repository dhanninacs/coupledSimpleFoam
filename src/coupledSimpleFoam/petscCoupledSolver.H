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

Class
    Foam::petscCoupledSolver

Description
    MPI backend for the coupled p-U system, built on PETSc MATMPIAIJ + KSP.

    Only compiled when the build found PETSc (-DHAVE_PETSC, see Make/options).

    Defaults: KSP fgmres with a fieldsplit-Schur preconditioner - bjacobi
    with sub-ILU on the velocity block, hypre BoomerAMG on the Schur
    complement - which requires a PETSc configured with hypre.  Everything
    can be overridden without recompiling, either through the PETSC_OPTIONS
    environment variable or through a petscOptions string in the COUPLED
    dictionary, e.g.

        COUPLED
        {
            linearSolver PETSc;
            petscOptions "-ksp_type fgmres -pc_type fieldsplit
                          -pc_fieldsplit_type schur";
        }

    This code is device agnostic.  MatSetFromOptions and VecSetFromOptions are
    called after the default types are set, so a build linked against a PETSc
    configured for a device runs the whole linear-algebra path there simply by
    naming device matrix and vector types in the same string, and the COO
    assembly below is already the PETSc device-assembly API and is unchanged.
    Nothing in this repository builds that configuration; a companion project
    supplies it and documents the options that go with it.

    Note that -use_gpu_aware_mpi and -log_view are consumed by PetscInitialize
    and therefore CANNOT be set here: the first is defaulted in the
    constructor, the second must come from PETSC_OPTIONS.

    See doc/USAGE.md for the runtime options.

SourceFiles
    petscCoupledSolver.C

\*---------------------------------------------------------------------------*/

#ifndef petscCoupledSolver_H
#define petscCoupledSolver_H

#ifdef HAVE_PETSC

#include "coupledLinearSystem.H"
#include "word.H"

namespace Foam
{

class petscCoupledSolver
:
    public coupledSolver
{
    // Private Data

        word name_;

        scalar tolerance_;
        scalar relTolerance_;
        label maxIter_;

        string petscOptions_;

        //- Opaque PETSc handles (Mat, Vec, Vec, KSP)
        void* matPtr_;
        void* rhsPtr_;
        void* solPtr_;
        void* kspPtr_;

        bool built_;

        //- Number of COO entries registered with MatSetPreallocationCOO
        //  (sparsity pattern must be static across outer iterations)
        label nCoo_;

        //- Freeze the preconditioner built on the first outer iteration
        bool reusePC_;

        //- Rebuild the preconditioner every pcReuseInterval_ solves and
        //  reuse it in between.  1 (the default) rebuilds every outer
        //  iteration, which is the historical behaviour.  Ignored when
        //  reusePC_ is set, which freezes it outright.
        label pcReuseInterval_;

        //- Number of solves performed (== outer iterations, since the outer
        //  loop calls solve() exactly once per iteration)
        label nSolves_;

        //- True once the outer loop has pushed an adaptive relative
        //  tolerance, which also selects the initial-residual-relative KSP
        //  convergence test (see setRelTol and solve)
        bool adaptiveRelTol_;


public:

    explicit petscCoupledSolver(const dictionary& dict);

    virtual ~petscCoupledSolver();

    virtual const word& name() const
    {
        return name_;
    }

    virtual label solve(const coupledLinearSystem& sys, scalarField& x);

    //- Relative KSP tolerance for the next solve (KSPSetTolerances rtol).
    //  Used by the adaptive (Eisenstat-Walker) inner tolerance in the outer
    //  loop; when adaptiveTolerance is off this is never called and the
    //  dictionary relTol stands for the whole run.
    //
    //  Being called at all also switches the KSP convergence test to the
    //  initial-residual norm, because that - not PETSc's default ||b|| - is
    //  the quantity an Eisenstat-Walker forcing term is defined against.
    //  See the comment in solve().
    virtual void setRelTol(const scalar rtol)
    {
        relTolerance_ = rtol;
        adaptiveRelTol_ = true;
    }
};


} // End namespace Foam

#endif // HAVE_PETSC

#endif

// ************************************************************************* //
