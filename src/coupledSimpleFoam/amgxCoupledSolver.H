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

Class
    Foam::amgxCoupledSolver

Description
    Serial GPU backend for the coupled p-U system on NVIDIA AmgX: the whole
    block-4 system is one 4x4 block-CSR matrix solved by a Krylov method
    preconditioned with block AMG (aggregation, multicolour block DILU
    smoothing by default).  No field split, so no submatrix extraction and
    none of the host round trips that come with it on PETSc's device path.

    Only compiled when the build found AmgX (-DHAVE_AMGX, see Make/options).
    Serial only: one rank, one GPU.

        COUPLED
        {
            linearSolver     AmgX;
            amgxPrecision    double;   // or mixed: double vectors, single-
                                       // precision matrix and preconditioner
            pcReuseInterval  8;        // rebuild the AMG hierarchy every N
            amgxReuse        none;     // between rebuilds: none | resetup
            amgxConfig       "...";    // optional: replaces the default
        }

    The outer solver's AmgX scope must be called "main": the relative
    tolerance and iteration cap are appended as main:tolerance and
    main:max_iters at every rebuild.  The tolerance in force is the one at
    the last rebuild, so the adaptive (Eisenstat-Walker) schedule moves in
    steps of pcReuseInterval.

SourceFiles
    amgxCoupledSolver.C

\*---------------------------------------------------------------------------*/

#ifndef amgxCoupledSolver_H
#define amgxCoupledSolver_H

#ifdef HAVE_AMGX

#include "coupledLinearSystem.H"
#include "word.H"

#include <cstdint>
#include <vector>

namespace Foam
{

class amgxCoupledSolver
:
    public coupledSolver
{
    // Private Data

        word name_;

        scalar relTolerance_;
        label maxIter_;
        label pcReuseInterval_;
        word reuse_;
        bool mixed_;
        bool verbose_;
        string config_;

        //- Opaque AmgX handles
        void* rsrc_;
        void* cfg_;
        void* solverCfg_;
        void* mat_;
        void* rhs_;
        void* sol_;
        void* solver_;

        //- Block pattern, built on the first solve
        label nBlockRows_;
        label nCoo_;
        std::vector<int> rowPtr_;
        std::vector<int> colIdx_;
        //- For each COO triplet, its slot in the block value array
        std::vector<int64_t> slot_;
        //- The triplets grouped by block row (counting sort), so the scatter
        //  runs in parallel over block rows with no write conflicts
        std::vector<int> byBlockRow_;
        std::vector<int64_t> blockRowFirst_;
        std::vector<double> valuesD_;
        std::vector<float> valuesF_;

        label nSolves_;
        bool uploaded_;


    // Private Member Functions

        void buildPattern(const coupledLinearSystem& sys);
        void fillValues(const coupledLinearSystem& sys);
        void rebuildSolver();


public:

    explicit amgxCoupledSolver(const dictionary& dict);

    virtual ~amgxCoupledSolver();

    virtual const word& name() const
    {
        return name_;
    }

    virtual label solve(const coupledLinearSystem& sys, scalarField& x);

    virtual void setRelTol(const scalar rtol)
    {
        relTolerance_ = rtol;
    }
};


} // End namespace Foam

#endif // HAVE_AMGX

#endif

// ************************************************************************* //
