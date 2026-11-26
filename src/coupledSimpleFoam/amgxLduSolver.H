/*---------------------------------------------------------------------------*\
  coupledSimpleFoam - Copyright (C) 2026 dhanninacs - GPL-3.0-or-later

Class
    Foam::amgxLduSolver

Description
    An OpenFOAM lduMatrix solver that solves a scalar equation on the GPU
    with NVIDIA AmgX: select it per field in fvSolution,

        T
        {
            solver      AmgX;
            tolerance   1e-12;
            relTol      0;
            maxIter     1000;
            amgxConfig  "...";      // optional, outer scope must be "main"
        }

    Residuals are OpenFOAM's own (normalisation factor and gSumMag, as in
    PBiCGStab), measured on the host before and after, so the log and
    residualControl mean what they always did.  The AmgX target is derived
    from tolerance/relTol and re-checked on the host; a solve that falls short
    of OpenFOAM's criterion is continued from where it stopped.

    The matrix pattern and the AmgX handles are cached per field, so later
    solves only replace coefficients and rebuild the AMG hierarchy.

    Serial only: a matrix with coupled (processor or cyclic) interfaces is a
    fatal error.

\*---------------------------------------------------------------------------*/

#ifndef amgxLduSolver_H
#define amgxLduSolver_H

#ifdef HAVE_AMGX

#include "lduMatrix.H"

namespace Foam
{

class amgxLduSolver
:
    public lduMatrix::solver
{
public:

    TypeName("AmgX");

    amgxLduSolver
    (
        const word& fieldName,
        const lduMatrix& matrix,
        const FieldField<Field, scalar>& interfaceBouCoeffs,
        const FieldField<Field, scalar>& interfaceIntCoeffs,
        const lduInterfaceFieldPtrsList& interfaces,
        const dictionary& solverControls
    );

    virtual ~amgxLduSolver()
    {}

    virtual solverPerformance solve
    (
        scalarField& psi,
        const scalarField& source,
        const direction cmpt = 0
    ) const;
};

} // End namespace Foam

#endif // HAVE_AMGX

#endif
