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
    Foam::coupledLinearSystem

Description
    Backend-agnostic sparse linear system for the fully coupled p-U solver,
    stored in COO (triplet) form.

    Row and column indices are GLOBAL and use the interleaved block layout

        index = blockSize*globalCellIndex + component      (u,v,w,p -> 0..3)

    where globalCellIndex comes from Foam::globalIndex over the mesh cells.
    In serial the global indices are identical to the local cell indices, so
    the same assembly code serves both the Eigen (serial) and PETSc (MPI)
    back-ends without modification.

    Each rank contributes only triplets whose ROW it owns; columns may point
    at cells owned by other ranks (they are reached through coupled - i.e.
    processor or cyclic - patches).

Class
    Foam::coupledSolver

Description
    Minimal interface for a linear-algebra backend:

        solve(system, x)  ->  x

    x is the local part of the solution, sized nLocalRows.

SourceFiles
    coupledLinearSystem.C

\*---------------------------------------------------------------------------*/

#ifndef coupledLinearSystem_H
#define coupledLinearSystem_H

#include "DynamicList.H"
#include "scalarField.H"
#include "labelList.H"
#include "Map.H"
#include "FixedList.H"
#include "dictionary.H"
#include "autoPtr.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{

/*---------------------------------------------------------------------------*\
                    Class coupledLinearSystem Declaration
\*---------------------------------------------------------------------------*/

class coupledLinearSystem
{
    // Private Data

        //- Unknowns per cell (4 for [u,v,w,p])
        const label blockSize_;

        //- Global index of this rank's first row
        const label rowStart_;

        //- Number of rows owned by this rank
        const label nLocalRows_;

        //- Number of rows in the global system
        const label nGlobalRows_;

        DynamicList<label> rows_;
        DynamicList<label> cols_;
        DynamicList<scalar> vals_;

        //- Right-hand side, local rows only
        scalarField rhs_;


public:

    // Constructors

        coupledLinearSystem
        (
            const label blockSize,
            const label rowStart,
            const label nLocalRows,
            const label nGlobalRows
        );


    // Member Functions

        // Access

            inline label blockSize() const { return blockSize_; }
            inline label rowStart() const { return rowStart_; }
            inline label nLocalRows() const { return nLocalRows_; }
            inline label nGlobalRows() const { return nGlobalRows_; }
            inline label nEntries() const { return rows_.size(); }

            inline label row(const label k) const { return rows_[k]; }
            inline label col(const label k) const { return cols_[k]; }
            inline scalar val(const label k) const { return vals_[k]; }

            //- Raw contiguous triplet arrays (for batched backends)
            inline const label* rowData() const { return rows_.cdata(); }
            inline const label* colData() const { return cols_.cdata(); }
            inline const scalar* valData() const { return vals_.cdata(); }

            inline const scalarField& rhs() const { return rhs_; }
            inline scalarField& rhs() { return rhs_; }

            //- True if the global row index is owned by this rank
            inline bool ownRow(const label gRow) const
            {
                return gRow >= rowStart_ && gRow < rowStart_ + nLocalRows_;
            }


        // Assembly

            //- Discard all entries, zero the RHS
            void clear();

            //- Pre-allocate space for nEntries triplets
            void reserve(const label n);

            //- Add a matrix entry (duplicates are summed by the backends)
            inline void add
            (
                const label gRow,
                const label gCol,
                const scalar v
            )
            {
                rows_.append(gRow);
                cols_.append(gCol);
                vals_.append(v);
            }

            //- Add to the right-hand side
            inline void addRhs(const label gRow, const scalar v)
            {
                rhs_[gRow - rowStart_] += v;
            }

            //- Replace a row by  1*x[gRow] = value  (used for the pressure
            //  reference cell).  Existing entries are zeroed in place; the
            //  resulting structural zeros are harmless for both backends.
            void setRowIdentity(const label gRow, const scalar value);


        // Linear algebra helpers

            //- result = A*x.  xLocal holds this rank's rows, xRemote the
            //  values of any column that belongs to another rank.
            void multiply
            (
                const scalarField& xLocal,
                const Map<scalar>& xRemote,
                scalarField& result
            ) const;

            //- result = A*xRef where xRef is uniform per block component
            void multiplyUniform
            (
                const FixedList<scalar, 4>& xRef,
                scalarField& result
            ) const;
};


/*---------------------------------------------------------------------------*\
                        Class coupledSolver Declaration
\*---------------------------------------------------------------------------*/

class coupledSolver
{
public:

    //- Runtime name of the backend actually used (for the log)
    virtual const word& name() const = 0;

    virtual ~coupledSolver()
    {}

    //- Solve; x is sized nLocalRows on entry and holds the initial guess.
    //  Returns the number of linear iterations performed (1 for a direct
    //  solve).
    virtual label solve
    (
        const coupledLinearSystem& sys,
        scalarField& x
    ) = 0;

    //- Set the relative (inner) tolerance to be used by the NEXT solve.
    //  Called by the outer loop only when adaptiveTolerance is switched on,
    //  which is why the default is a no-op: a backend with no relative
    //  tolerance to set - the Eigen direct solve - simply ignores the
    //  request and the outer loop is unaffected.  See doc/USAGE.md 2.2.
    virtual void setRelTol(const scalar)
    {}

    //- Select a backend.  "SparseLU" / "BiCGSTAB" are Eigen (serial),
    //  "PETSc" is the MPI backend.  Defaults to PETSc for a parallel run
    //  and SparseLU otherwise.
    static autoPtr<coupledSolver> New(const dictionary& dict);
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
