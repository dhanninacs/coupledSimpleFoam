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
    Foam::eigenCoupledSolver

Description
    Serial Eigen backend for the coupled p-U system.

    linearSolver SparseLU   sparse direct LU with COLAMD ordering (default)
    linearSolver BiCGSTAB   BiCGSTAB preconditioned with IncompleteLUT

    The BiCGSTAB path works on a row-major matrix, so Eigen runs its
    matrix-vector products on all OpenMP threads; keeps its IncompleteLUT
    between outer iterations and rebuilds it every pcReuseInterval solves
    (default 1: every solve); and takes the adaptive inner tolerance
    (adaptiveTolerance, Eisenstat-Walker) through setRelTol, as the PETSc and
    AmgX backends do.

SourceFiles
    eigenCoupledSolver.C

\*---------------------------------------------------------------------------*/

#ifndef eigenCoupledSolver_H
#define eigenCoupledSolver_H

#include "coupledLinearSystem.H"
#include "word.H"

namespace Foam
{

class eigenCoupledSolver
:
    public coupledSolver
{
    // Private Data

        word name_;

        //- BiCGSTAB controls
        scalar tolerance_;
        label maxIter_;
        scalar dropTolerance_;
        label fillFactor_;

        //- SparseLU keeps its symbolic factorisation between outer
        //  iterations (the sparsity pattern never changes)
        bool patternAnalysed_;

        //- Opaque handle to the Eigen SparseLU object
        void* luPtr_;

        //- BiCGSTAB: the kept IncompleteLUT (opaque), how often it is
        //  rebuilt, how many solves since, and the adaptive tolerance
        void* ilutPtr_;
        label pcReuseInterval_;
        label solvesSinceBuild_;
        scalar relTol_;


public:

    eigenCoupledSolver(const dictionary& dict, const word& backend);

    virtual ~eigenCoupledSolver();

    virtual const word& name() const
    {
        return name_;
    }

    virtual label solve(const coupledLinearSystem& sys, scalarField& x);

    //- The inner tolerance of the next BiCGSTAB solve (adaptiveTolerance)
    virtual void setRelTol(const scalar rtol)
    {
        relTol_ = rtol;
    }
};


} // End namespace Foam

#endif

// ************************************************************************* //
