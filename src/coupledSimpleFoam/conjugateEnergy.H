/*---------------------------------------------------------------------------*\
  coupledSimpleFoam - Copyright (C) 2026 dhanninacs - GPL-3.0-or-later

Class
    Foam::conjugateEnergy

Description
    Conjugate heat transfer for coupledSimpleFoam: the fluid region's
    temperature and every solid region's temperature solved as ONE linear
    system, the interfaces written into the matrix rather than exchanged as
    boundary values between region solves.

    Run the solver on the fluid region (-region fluid) and name the solids:

        COUPLED
        {
            solveEnergy     yes;
            Pr              0.7;
            solidRegions    (solid);      // regions from splitMeshRegions
            rhoCp           1;            // fluid rho*Cp [J/m^3/K]
            conjugateSolver AmgX;         // AmgX (GPU) or Eigen (CPU)
            conjugateTolerance 1e-10;     // OpenFOAM-normalised residual
            conjugateMaxIter   1000;
        }

    Each region's equation is OpenFOAM's own fvScalarMatrix: the fluid's
    div(phi,T) - laplacian(alphaEff,T), scaled by rhoCp so it is in W/K like
    the solids' -laplacian(kappa,T) (kappa from constant/<solid>/
    physicalProperties, mixture/transport/kappa).  Boundary coefficients of
    every non-interface patch are added as fvMatrix::solve adds them.  An
    interface face pair (fluid patch <fluid>_to_<solid>, solid patch
    <solid>_to_<fluid>, matched by face centre) adds the series conductance
        G = |Sf| / (1/(k_f*delta_f) + 1/(k_s*delta_s))
    between its two cells - the discrete form of continuous temperature and
    heat flux at the interface, which is what coupledTemperature converges
    to.  After the solve the interface patches (fixedValue in the case) are
    set to the interface temperature, so post-processing sees it.

    Serial only (one rank), conformal interfaces only (every face matched).

\*---------------------------------------------------------------------------*/

#ifndef conjugateEnergy_H
#define conjugateEnergy_H

#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "PtrList.H"

#include <memory>
#include <vector>

namespace Foam
{

class conjugateEnergy
{
    struct interface
    {
        label solid;
        label fluidPatch;
        label solidPatch;
        //- Matched faces: fluidFace[i] (in the fluid patch) <-> solidFace[i]
        labelList fluidFace;
        labelList solidFace;
    };

    fvMesh& fluid_;
    volScalarField& Tf_;
    const scalar rhoCp_;
    const word solverName_;
    const scalar tolerance_;
    const label maxIter_;

    PtrList<fvMesh> solids_;
    PtrList<volScalarField> Ts_;
    scalarList kappa_;
    List<interface> ifaces_;

    //- Row offset of each region (0 = fluid, 1.. = solids) and the total
    labelList offset_;
    label n_;

    //- CSR pattern of the combined system and, for every contribution in
    //  assembly order, its slot in the value array
    std::vector<int> rowPtr_;
    std::vector<int> colIdx_;
    std::vector<int> slot_;
    std::vector<double> values_;
    std::vector<double> rhs_;
    std::vector<double> x_;
    bool patternBuilt_;

    //- Opaque AmgX state (matrix, vectors) when the GPU solver is used
    void* amgx_;

    void matchInterfaces();
    void buildPattern();
    //- Contributions in assembly order (must match buildPattern)
    void assemble
    (
        const surfaceScalarField& phi,
        const volScalarField& alphaEff,
        std::vector<double>& contrib
    );
    label solveSystem(scalar initialResidual, scalar normFactor);
    void residual(scalar& res, scalar& normFactor) const;

public:

    conjugateEnergy
    (
        fvMesh& fluid,
        volScalarField& Tf,
        const dictionary& coupledDict
    );

    ~conjugateEnergy();

    //- Assemble, solve and distribute T over all regions; records the
    //  residual for residualControl like fvMatrix::solve does
    void solve(const surfaceScalarField& phi, const volScalarField& alphaEff);
};

} // End namespace Foam

#endif
