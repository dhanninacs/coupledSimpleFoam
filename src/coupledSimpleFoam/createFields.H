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
    viscosity model, the momentum-transport model, the rAU field that
    carries the Rhie-Chow coefficient V/aP, and - only when the COUPLED
    dictionary asks for it - the temperature T.

    Requires coupledDict, the COUPLED sub-dictionary of system/fvSolution,
    to have been read already: solveEnergy decides whether T exists at all.

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

// ------------------------------------------------------------------ energy
// Temperature transport is optional and off by default, and T is CONSTRUCTED
// ONLY when it is on: a case that carries no solveEnergy entry needs no 0/T,
// registers no extra field and follows exactly the same code path as before
// the feature existed.
//
// For constant-property incompressible flow with no buoyancy T is a PASSIVE
// scalar - it takes phi and nut from the flow and returns nothing to the
// momentum rows - so the coupled p-U assembly is identical whether or not
// this block runs.  doc/USAGE.md 9 states the assumption and its limits.
const Switch solveEnergy
(
    coupledDict.lookupOrDefault<Switch>("solveEnergy", Switch(false))
);

// Pr and Prt sit in the COUPLED dictionary with every other control of this
// solver, not in constant/physicalProperties.  Three reasons: that file is a
// run-time-selection dictionary owned by viscosityModel, so a key it does not
// know is read by nothing and checked by nobody; OpenFOAM's own passive-scalar
// mechanism, the scalarTransport functionObject, likewise keeps its alphal and
// alphat with the transport controls rather than with the physical properties;
// and keeping the three keys together means enabling heat transfer is one
// edit in one block.  A genuinely thermal (compressible, buoyant) extension
// would take Pr from a thermophysicalProperties dictionary instead, because
// there it is a property of a fluid model rather than a coefficient of a
// passive scalar.
const dimensionedScalar Pr
(
    "Pr",
    dimless,
    coupledDict.lookupOrDefault<scalar>("Pr", 0.71)
);

const dimensionedScalar Prt
(
    "Prt",
    dimless,
    coupledDict.lookupOrDefault<scalar>("Prt", 0.85)
);

autoPtr<volScalarField> TPtr;

if (solveEnergy)
{
    if (Pr.value() <= 0 || Prt.value() <= 0)
    {
        FatalErrorInFunction
            << "Pr (" << Pr.value() << ") and Prt (" << Prt.value()
            << ") in the COUPLED dictionary must both be positive"
            << exit(FatalError);
    }

    // MUST_READ on its own dies inside the field read with a message about a
    // missing file that never mentions solveEnergy, so probe the header first
    // - NO_READ, so that the probe itself cannot raise that error - and say
    // what the case is actually missing.  headerOk() also checks the class
    // name, so a 0/T that is not a volScalarField is caught here too.
    typeIOobject<volScalarField> THeader
    (
        "T",
        runTime.name(),
        mesh,
        IOobject::NO_READ,
        IOobject::NO_WRITE
    );

    if (!THeader.headerOk())
    {
        FatalErrorInFunction
            << "solveEnergy is on in the COUPLED dictionary of"
            << " system/fvSolution, but the case has no readable"
            << " temperature field " << runTime.name() << "/T." << nl
            << "Add a T field carrying the thermal boundary conditions"
            << " (fixedValue for an isothermal wall, zeroGradient for an"
            << " adiabatic one, fixedGradient for a prescribed heat flux,"
            << " mixed for an external resistance), or set solveEnergy no."
            << exit(FatalError);
    }

    TPtr.set
    (
        new volScalarField
        (
            IOobject
            (
                "T",
                runTime.name(),
                mesh,
                IOobject::MUST_READ,
                IOobject::AUTO_WRITE
            ),
            mesh
        )
    );

    Info<< "Energy transport: solving T segregated after the coupled p-U"
        << " system," << nl
        << "    alphaEff = nu/Pr + nut/Prt with Pr = " << Pr.value()
        << ", Prt = " << Prt.value() << nl << endl;
}
