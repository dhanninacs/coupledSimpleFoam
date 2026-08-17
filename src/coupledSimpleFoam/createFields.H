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
    Field and model construction for coupledSimpleFoam: p, U, phi, the
    viscosity model, the momentum-transport model and the rAU field that
    carries the Rhie-Chow coefficient V/aP.

\*---------------------------------------------------------------------------*/

Info<< "Reading field p\n" << endl;
volScalarField p
(
    IOobject
    (
        "p",
        runTime.name(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

Info<< "Reading field U\n" << endl;
volVectorField U
(
    IOobject
    (
        "U",
        runTime.name(),
        mesh,
        IOobject::MUST_READ,
        IOobject::AUTO_WRITE
    ),
    mesh
);

#include "createPhi.H"

// Viscosity and momentum-transport models, constructed exactly as the stock
// OpenFOAM 14 incompressibleFluid solver module constructs them
// (applications/modules/incompressibleFluid/incompressibleFluid.C):
//
//   viscosity        <- constant/physicalProperties  ("viscosityModel <name>")
//                       falls back to thermophysicalProperties and then to
//                       transportProperties, handled by
//                       physicalProperties::findModelDict
//   momentumTransport<- constant/momentumTransport   ("simulationType <name>")
//                       falls back to turbulenceProperties.
//
// The momentum-transport model owns nut/k/epsilon/omega and every wall
// function attached to them, so laminar / kEpsilon / kOmegaSST / realizableKE
// / Spalart-Allmaras / ... all become available without any further code.
Info<< "Constructing viscosity and momentum transport models" << nl << endl;

autoPtr<viscosityModel> viscosity(viscosityModel::New(mesh));

autoPtr<incompressible::momentumTransportModel> momentumTransport
(
    incompressible::momentumTransportModel::New(U, phi, viscosity)
);

// Update the derived fields (nut from the wall functions for a RAS model,
// a no-op for laminar) before the first momentum matrix is assembled.
momentumTransport->validate();

// V/aP, aP being the coupled momentum diagonal.  zeroGradient so that the
// patch value is the owner-cell value and coupled (processor/cyclic)
// patches are exchanged by correctBoundaryConditions().
volScalarField rAU
(
    IOobject
    (
        "rAU",
        runTime.name(),
        mesh
    ),
    mesh,
    dimensionedScalar("rAU", dimTime, 1.0),
    zeroGradientFvPatchScalarField::typeName
);
