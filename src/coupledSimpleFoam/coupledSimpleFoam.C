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

Application
    coupledSimpleFoam

Description
    Steady-state solver for incompressible laminar or RANS-turbulent flow that
    solves the momentum and continuity equations as ONE fully coupled linear
    system (block size 4, unknowns interleaved per cell as [u, v, w, p]).
    Turbulence is treated in the standard production way: segregated and
    lagged one outer iteration behind the coupled p-U solve.

    Assembly (executable specification: the independent 2-D reference
    implementation in tools/reference_prototype/coupled_vs_simple.py,
    function run_coupled):

    1.  Momentum blocks from an UNRELAXED
            fvVectorMatrix UEqn
            (
                fvm::div(phi, U) + momentumTransport->divDevSigma(U)
            )
        i.e. exactly the operator that the stock incompressibleFluid solver
        module builds (minus ddt/MRF/fvModels), using diag()/upper()/lower()
        with lduAddr().lowerAddr()/upperAddr(), plus internalCoeffs() on the
        diagonal and boundaryCoeffs() on the RHS (or as the off-diagonal link
        to the remote cell on coupled patches).

    2.  Implicit Green-Gauss pressure gradient in the momentum rows, built
        with the mesh interpolation weights and mesh.Sf():
            row(u_P) += Sf.x*(w p_P + (1-w) p_N),  row(u_N) -= same
        Boundary patches use the p field's own valueInternalCoeffs() /
        valueBoundaryCoeffs(), so fixedValue-p goes to the RHS and
        zeroGradient-p lands on the diagonal p column of the owner cell.

    3.  Continuity row = sum of the Rhie-Chow face fluxes:
            phi_f = (w U_P + (1-w) U_N) & Sf
                  - Df_f |Sf| Delta_f (p_N - p_P)          [implicit]
                  + Df_f (grad(p)_f & Seff_f)              [lagged -> RHS]
        with Df = interpolate(V/aP) and Delta = nonOrthDeltaCoeffs.  The
        non-orthogonal correction stays explicit, as in the standard pEqn.
        Patches where U fixes the value contribute their known flux to the
        RHS only; walls contribute nothing.

    4.  Optional pseudo-transient continuation (V/dtau on the diagonal,
        V/dtau*U_old on the RHS), default off.

    5.  After the solve, phi is rebuilt with exactly the same expression and
        the same lagged gradient, so the discrete continuity rows are
        satisfied to machine precision.

    6.  With phi current, viscosity->correct() and momentumTransport->correct()
        run the segregated turbulence update: the k/epsilon/omega/nuTilda
        transport equations are solved with the fvSolution "solvers" and
        "relaxationFactors" entries, exactly as SIMPLE does, and their
        residual lines appear in the log in the standard format.

    7.  Optionally (COUPLED/solveEnergy, default off) a temperature equation
            fvm::div(phi, T) - fvm::laplacian(alphaEff, T)
        with alphaEff = nu/Pr + nut/Prt is solved segregated on that same
        conservative flux, in the same way and with the same relaxation and
        residual-control machinery as the turbulence equations.  For constant
        properties and no buoyancy T is a PASSIVE scalar: it never feeds back
        into the momentum rows, so with the switch off - or on - the coupled
        p-U assembly above is bit-for-bit the same.

    The linear system is assembled backend-agnostically as COO triplets with
    GLOBAL row/column indices (4*globalCell + component), so the same
    assembly feeds the serial Eigen backend and the MPI PETSc backend.

\*---------------------------------------------------------------------------*/

#include <cmath>
#include <chrono>
#include <string>

// OpenFOAM 14 no longer ships the fvCFD.H umbrella header
#include "argList.H"
#include "Time.H"
#include "fvMesh.H"
#include "volFields.H"
#include "surfaceFields.H"
#include "fvm.H"
#include "fvc.H"
#include "fvMatrices.H"
#include "IOdictionary.H"
#include "dimensionedScalar.H"
#include "Switch.H"
#include "zeroGradientFvPatchFields.H"
#include "globalIndex.H"
#include "syncTools.H"
#include "OSspecific.H"
#include "symmetryFvPatch.H"
#include "symmetryPlaneFvPatch.H"
#include "wedgeFvPatch.H"
#include "wallFvPatch.H"
#include "slipFvPatchFields.H"
#include "viscosityModel.H"
#include "incompressibleMomentumTransportModel.H"
#include "convergenceControl.H"
#include "coupledLinearSystem.H"
#include "conjugateEnergy.H"
#include "amgxCommon.H"
#include <cstdlib>

using namespace Foam;

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

int main(int argc, char *argv[])
{
    #ifdef HAVE_AMGX
    // Start CUDA (through AmgX) before anything else happens in the process:
    // before OpenFOAM installs its signal handlers, reads the mesh or starts
    // OpenMP threads, and while the address space is still empty. On one
    // Windows Server 2025 host (driver 582.16) the driver's cuInit faulted
    // when AmgX started it at the first linear solve (FoamWin issue #1).
    // COUPLED_AMGX_LAZY=1 keeps the old start, at the first AmgX solve.
    const char* lazy = std::getenv("COUPLED_AMGX_LAZY");
    const bool amgxEarly = !(lazy && lazy[0] && lazy[0] != '0');
    if (amgxEarly)
    {
        Foam::amgxCommon::acquire();
    }
    #endif

    #include "addRegionOption.H"
    #include "setRootCase.H"
    #include "createTime.H"
    // The mesh of -region <name> (the fluid region of a conjugate case), else
    // the default region: createMesh.H with the region name of setRegionName.H.
    #include "setRegionName.H"
    Foam::fvMesh mesh
    (
        Foam::IOobject(regionName, runTime.name(), runTime, Foam::IOobject::MUST_READ)
    );

    // ------------------------------------------------------------ controls
    // Read before the fields, because COUPLED/solveEnergy decides whether the
    // temperature field is constructed at all.
    const dictionary coupledDict
    (
        mesh.solution().found("COUPLED")
      ? mesh.solution().subDict("COUPLED")
      : dictionary::null
    );

    #include "createFields.H"

    // Conjugate heat transfer: the solid regions named in COUPLED/solidRegions
    // and the fluid's temperature solved as one system (conjugateEnergy.H).
    autoPtr<conjugateEnergy> conjugate;
    if (solveEnergy && coupledDict.found("solidRegions"))
    {
        conjugate.reset(new conjugateEnergy(mesh, TPtr(), coupledDict));
    }

    const label nOuterIterations =
        coupledDict.lookupOrDefault<label>("nOuterIterations", 100);

    const Switch pseudoTransient
    (
        coupledDict.lookupOrDefault<Switch>("pseudoTransient", Switch(false))
    );
    const scalar pseudoCo =
        coupledDict.lookupOrDefault<scalar>("pseudoCo", 200.0);

    // Scale factor on the Rhie-Chow D_f = V/aP.  1 is the natural coupled
    // value (no relaxation exists in a coupled solve).  Segregated SIMPLE
    // implicitly uses alpha_U*V/aP because rAU = 1/UEqn.A() is taken from
    // the RELAXED momentum matrix, so setting this to the reference run's
    // momentum relaxation factor reproduces the segregated discretisation
    // exactly - useful for verification, not for production.
    const scalar rcFactor =
        coupledDict.lookupOrDefault<scalar>("rhieChowFactor", 1.0);

    // ------------------------------- adaptive inner tolerance (Eisenstat-
    // Walker).  The inner Krylov solve does not need to be accurate while
    // the outer Picard iteration is still far from the solution: the forcing
    // term below is loose early and tightens as the NONLINEAR residual falls,
    // which is what avoids both failure modes documented in USAGE 4.1 - a
    // fixed loose tolerance plateaus the outer loop, a fixed tight one makes
    // every outer iteration unaffordable.  Off by default, so an existing
    // case is bit-for-bit unaffected: with the switch off no tolerance is
    // ever pushed to the backend and the dictionary relTol stands alone.
    const Switch adaptiveTolerance
    (
        coupledDict.lookupOrDefault<Switch>("adaptiveTolerance", Switch(false))
    );
    const scalar relTolMax =
        coupledDict.lookupOrDefault<scalar>("relTolMax", 1e-1);
    const scalar relTolMin =
        coupledDict.lookupOrDefault<scalar>("relTolMin", 1e-8);
    const scalar adaptiveGamma =
        coupledDict.lookupOrDefault<scalar>("adaptiveGamma", 0.9);
    const scalar adaptiveAlpha =
        coupledDict.lookupOrDefault<scalar>("adaptiveAlpha", 1.5);

    if (adaptiveTolerance)
    {
        if (relTolMin > relTolMax)
        {
            FatalErrorInFunction
                << "relTolMin (" << relTolMin << ") exceeds relTolMax ("
                << relTolMax << ")" << exit(FatalError);
        }

        Info<< "Adaptive inner tolerance (Eisenstat-Walker): relTol in ["
            << relTolMin << ", " << relTolMax << "], gamma "
            << adaptiveGamma << ", alpha " << adaptiveAlpha << nl << endl;
    }

    scalar resControlU = 1e-5;
    scalar resControlP = 1e-4;

    // Any residualControl key that is not U or p is taken to be a segregated
    // field - a turbulence field solved by momentumTransport->correct(), or T
    // when solveEnergy is on.  Its initial residual is read back from
    // OpenFOAM's own per-time-step residual store through convergenceControl,
    // which is what SIMPLE/PIMPLE use, so the criterion means exactly the same
    // thing as in the segregated solver.  Fields that are never solved are
    // simply not checked (again as in convergenceControl::criteriaSatisfied),
    // which is why a stale `T 1e-6` left in a case with solveEnergy off is
    // inert rather than a run that can never converge.
    DynamicList<word> extraResName;
    DynamicList<scalar> extraResTol;

    if (coupledDict.found("residualControl"))
    {
        const dictionary& rc = coupledDict.subDict("residualControl");
        resControlU = rc.lookupOrDefault<scalar>("U", 1e-5);
        resControlP = rc.lookupOrDefault<scalar>("p", 1e-4);

        const wordList rcKeys(rc.toc());
        forAll(rcKeys, i)
        {
            if (rcKeys[i] == "U" || rcKeys[i] == "p")
            {
                continue;
            }

            extraResName.append(rcKeys[i]);
            extraResTol.append(rc.lookup<scalar>(rcKeys[i]));
        }
    }

    // -1 => not measured yet (field not solved, or first outer iteration)
    scalarField extraRes(extraResName.size(), -1.0);

    if (extraResName.size())
    {
        Info<< "Additional (segregated) residual controls:";
        forAll(extraResName, i)
        {
            Info<< ' ' << extraResName[i] << ' ' << extraResTol[i];
        }
        Info<< nl << endl;
    }

    // ---------------------------------------------------- global indexing
    const globalIndex globalCells(mesh.nCells());

    const label blockSize = 4;
    const label rowStart = blockSize*globalCells.toGlobal(0);
    const label nLocalRows = blockSize*mesh.nCells();
    const label nGlobalRows = blockSize*globalCells.size();

    labelList gCell(mesh.nCells());
    forAll(gCell, celli)
    {
        gCell[celli] = globalCells.toGlobal(celli);
    }

    // Global cell index on the far side of every boundary face (valid for
    // coupled - processor / cyclic - patches; own cell elsewhere)
    labelList nbrGCell;
    syncTools::swapBoundaryCellList(mesh, gCell, nbrGCell);

    // Impermeable constraint patches (symmetry planes, wedges, slip walls)
    // carry EXACTLY zero mass flux.  Their per-component transform
    // coefficients cannot express that on tilted (snappyHexMesh) faces -
    // the diagonal-only representation leaks mass through the plane - so
    // the continuity row and the flux reconstruction treat them like
    // walls.
    auto zeroFluxPatch = [&](const label patchi) -> bool
    {
        const fvPatch& zfp = mesh.boundary()[patchi];
        return
            isA<symmetryFvPatch>(zfp)
         || isA<symmetryPlaneFvPatch>(zfp)
         || isA<wedgeFvPatch>(zfp)
         || isA<slipFvPatchVectorField>(U.boundaryField()[patchi]);
    };

    // --------------------------------------------------- pressure level
    bool pFixedByPatch = false;
    word pFixedPatchName = word::null;
    forAll(p.boundaryField(), patchi)
    {
        if (p.boundaryField()[patchi].fixesValue())
        {
            pFixedByPatch = true;
            pFixedPatchName = mesh.boundary()[patchi].name();
            break;
        }
    }
    {
        label flag = pFixedByPatch ? 1 : 0;
        reduce(flag, maxOp<label>());
        pFixedByPatch = (flag == 1);
    }

    const label pRefCell = coupledDict.lookupOrDefault<label>("pRefCell", 0);
    const scalar pRefValue =
        coupledDict.lookupOrDefault<scalar>("pRefValue", 0.0);

    // pRefCell is a GLOBAL cell index (cell 0 lives on rank 0)
    const label pinRow = blockSize*pRefCell + (blockSize - 1);
    const bool pinPressure = !pFixedByPatch;

    if (pinPressure)
    {
        Info<< "Pressure level: pinned at global cell " << pRefCell
            << " to p = " << pRefValue << nl << endl;
    }
    else
    {
        Info<< "Pressure level: set by fixedValue patch "
            << pFixedPatchName << nl << endl;
    }

    // ------------------------------------------------- empty directions
    const Vector<label> solD(mesh.solutionD());
    bool solveComp[3];
    for (label c = 0; c < 3; c++)
    {
        solveComp[c] = (solD[c] != -1);
    }

    // ------------------------------------------- face geometry constants
    // Seff_f is the effective area vector of the lagged Rhie-Chow term:
    //     Seff = |Sf| * Delta * d      (== Sf on an orthogonal mesh)
    // so that  implicit orthogonal Laplacian + lagged (grad(p).Seff) is the
    // consistent split of the full Rhie-Chow pressure term.
    surfaceVectorField Seff
    (
        IOobject("Seff", runTime.name(), mesh),
        mesh,
        dimensionedVector("Seff", dimArea, vector::zero)
    );
    {
        const surfaceScalarField& magSf = mesh.magSf();
        const surfaceScalarField& nodc = mesh.nonOrthDeltaCoeffs();
        const volVectorField& CC = mesh.C();
        const labelUList& own = mesh.owner();
        const labelUList& nei = mesh.neighbour();

        forAll(own, facei)
        {
            Seff[facei] =
                magSf[facei]*nodc[facei]*(CC[nei[facei]] - CC[own[facei]]);
        }

        forAll(mesh.boundary(), patchi)
        {
            const fvPatch& fvp = mesh.boundary()[patchi];
            const vectorField d(fvp.delta());

            if (fvp.coupled())
            {
                Seff.boundaryFieldRef()[patchi] =
                    magSf.boundaryField()[patchi]
                   *nodc.boundaryField()[patchi]*d;
            }
            else
            {
                Seff.boundaryFieldRef()[patchi] =
                    magSf.boundaryField()[patchi]*fvp.deltaCoeffs()*d;
            }
        }
    }

    // --------------------------------------------------- linear algebra
    coupledLinearSystem sys(blockSize, rowStart, nLocalRows, nGlobalRows);
    autoPtr<coupledSolver> lsolver(coupledSolver::New(coupledDict));

    Info<< "Coupled p-U system: " << nGlobalRows << " unknowns, backend "
        << lsolver->name() << nl << endl;

    scalarField xOld(nLocalRows, 0.0);
    scalarField xNew(nLocalRows, 0.0);
    scalarField Ax(nLocalRows, 0.0);
    scalarField AxRef(nLocalRows, 0.0);
    Map<scalar> xRemote;

    scalar cumulativeContErr = 0.0;

    // Pack the current fields into the local solution vector
    auto packX = [&](scalarField& xv)
    {
        forAll(U, celli)
        {
            xv[blockSize*celli + 0] = U[celli].x();
            xv[blockSize*celli + 1] = U[celli].y();
            xv[blockSize*celli + 2] = U[celli].z();
            xv[blockSize*celli + 3] = p[celli];
        }
    };

    // Values of the cells owned by other ranks, keyed by global row index
    auto buildRemote = [&]()
    {
        xRemote.clear();

        List<vector> nbrU;
        List<scalar> nbrP;
        syncTools::swapBoundaryCellList(mesh, U.primitiveField(), nbrU);
        syncTools::swapBoundaryCellList(mesh, p.primitiveField(), nbrP);

        forAll(mesh.boundary(), patchi)
        {
            const fvPatch& fvp = mesh.boundary()[patchi];

            if (!fvp.coupled() || fvp.size() == 0)
            {
                continue;
            }

            const label bStart = fvp.start() - mesh.nInternalFaces();

            forAll(fvp, i)
            {
                const label g = nbrGCell[bStart + i];

                if (sys.ownRow(blockSize*g))
                {
                    continue;
                }

                xRemote.set(blockSize*g + 0, nbrU[bStart + i].x());
                xRemote.set(blockSize*g + 1, nbrU[bStart + i].y());
                xRemote.set(blockSize*g + 2, nbrU[bStart + i].z());
                xRemote.set(blockSize*g + 3, nbrP[bStart + i]);
            }
        }
    };

    // A velocity component that carries next to none of the flow (the cross-stream
    // component of a periodic or symmetric case) has a normalisation factor made of
    // round-off, so its OpenFOAM-normalised residual stays O(1) and residualControl
    // never fires although the flow is converged. Each velocity component's factor is
    // floored at residualFloor times the largest velocity component's factor.
    const scalar residualFloor =
        coupledDict.lookupOrDefault<scalar>("residualFloor", 1e-3);

    // OpenFOAM-style normalised residual, per block component
    auto blockResiduals = [&](const scalarField& xv, FixedList<scalar, 4>& res)
    {
        sys.multiply(xv, xRemote, Ax);

        FixedList<scalar, 4> xRef(0.0);
        for (label c = 0; c < blockSize; c++)
        {
            scalar s = 0.0;
            for (label celli = 0; celli < mesh.nCells(); celli++)
            {
                s += xv[blockSize*celli + c];
            }
            reduce(s, sumOp<scalar>());
            xRef[c] = s/max(scalar(globalCells.size()), 1.0);
        }

        sys.multiplyUniform(xRef, AxRef);

        FixedList<scalar, 4> num(0.0);
        FixedList<scalar, 4> den(0.0);

        for (label celli = 0; celli < mesh.nCells(); celli++)
        {
            for (label c = 0; c < blockSize; c++)
            {
                const label i = blockSize*celli + c;
                num[c] += mag(sys.rhs()[i] - Ax[i]);
                den[c] +=
                    mag(Ax[i] - AxRef[i]) + mag(sys.rhs()[i] - AxRef[i]);
            }
        }

        for (label c = 0; c < blockSize; c++)
        {
            reduce(num[c], sumOp<scalar>());
            reduce(den[c], sumOp<scalar>());
        }
        const scalar denU = max(max(den[0], den[1]), den[2]);
        for (label c = 0; c < blockSize; c++)
        {
            const scalar d = c < 3 ? max(den[c], residualFloor*denU) : den[c];
            res[c] = num[c]/max(d, vSmall);
        }
    };

    const word cmptName[4] = {"Ux", "Uy", "Uz", "p"};

    Info<< "\nStarting coupled iteration loop\n" << endl;

    label iter = 0;
    bool converged = false;

    // Eisenstat-Walker state: the nonlinear residual and the forcing term of
    // the previous outer iteration.  Negative means "not measured yet".
    scalar ewResPrev = -1.0;
    scalar ewEtaPrev = -1.0;

    const bool timing = coupledDict.lookupOrDefault<bool>("timing", false);
    auto timerLast = std::chrono::steady_clock::now();
    std::string timerLine;
    auto sectionTimer = [&](const char* name)
    {
        const auto now = std::chrono::steady_clock::now();
        if (timing)
        {
            timerLine += std::string(" ") + name + " "
                + std::to_string(std::chrono::duration<double>(now - timerLast).count());
        }
        timerLast = now;
    };

    while (iter < nOuterIterations)
    {
        iter++;
        sectionTimer("energy+rest");
        if (timing && !timerLine.empty())
        {
            Info<< "timing:" << timerLine.c_str() << endl;
        }
        timerLine.clear();
        timerLast = std::chrono::steady_clock::now();
        runTime++;

        Info<< "Time = " << runTime.userTimeName() << nl << endl;

        // ------------------------------------------- momentum operator
        // Identical to the stock incompressibleFluid momentum predictor
        // (applications/modules/incompressibleFluid/momentumPredictor.C)
        // without ddt (steady), MRF and fvModels, and crucially UNRELAXED -
        // a coupled solve has no equation relaxation.  For a laminar
        // (Stokes) model divDevSigma expands to
        //     fvm::divc(-nuEff_f*(Sf & dev2(T(grad(U)))), U)
        //   - fvm::laplacian(nuEff_f, U)
        // i.e. exactly the old "-fvm::laplacian(nuf,U) + divDevTauCorrection"
        // pair; for a RAS model nuEff = nu + nut and the whole turbulence
        // machinery (wall functions included) comes with it.
        fvVectorMatrix UEqn
        (
            fvm::div(phi, U)
          + momentumTransport->divDevSigma(U)
        );

        // --------------------------- component-wise coupled diagonal aP
        vectorField aP(mesh.nCells(), vector::zero);
        {
            const scalarField& diag = UEqn.diag();
            forAll(aP, celli)
            {
                aP[celli] = vector(diag[celli], diag[celli], diag[celli]);
            }

            forAll(mesh.boundary(), patchi)
            {
                const labelUList& fc = mesh.boundary()[patchi].faceCells();
                const vectorField& ic = UEqn.internalCoeffs()[patchi];

                forAll(fc, i)
                {
                    aP[fc[i]] += ic[i];
                }
            }
        }

        // ------------------------------- pseudo-transient continuation
        scalarField dtauDiag(mesh.nCells(), 0.0);
        if (pseudoTransient)
        {
            scalar Uref = coupledDict.lookupOrDefault<scalar>("pseudoUref", 0);

            if (Uref <= 0)
            {
                Uref = small;

                forAll(U, celli)
                {
                    Uref = max(Uref, mag(U[celli]));
                }

                forAll(U.boundaryField(), patchi)
                {
                    const fvPatchVectorField& pU = U.boundaryField()[patchi];
                    forAll(pU, i)
                    {
                        Uref = max(Uref, mag(pU[i]));
                    }
                }

                reduce(Uref, maxOp<scalar>());
            }

            forAll(dtauDiag, celli)
            {
                const scalar h = std::cbrt(mesh.V()[celli]);
                const scalar Umag =
                    max(mag(U[celli]), 0.05*Uref);
                const scalar dtau = pseudoCo*h/Umag;

                dtauDiag[celli] = mesh.V()[celli]/dtau;
                aP[celli] += vector(1, 1, 1)*dtauDiag[celli];
            }
        }

        // ------------------------------------------------ Rhie-Chow D_f
        forAll(rAU, celli)
        {
            rAU[celli] = rcFactor*mesh.V()[celli]/cmptAv(aP[celli]);
        }
        rAU.correctBoundaryConditions();

        const surfaceScalarField Df("Df", fvc::interpolate(rAU));

        // ------------------------------------- lagged pressure gradient
        const volVectorField gradP(fvc::grad(p));
        const surfaceVectorField gradPf(fvc::interpolate(gradP));

        sectionTimer("operators");
        // ==================================================== ASSEMBLY
        sys.clear();
        sys.reserve
        (
            blockSize*blockSize*(mesh.nCells() + 2*mesh.nInternalFaces())
        );

        {
            const surfaceScalarField& w = mesh.weights();
            const surfaceVectorField& Sf = mesh.Sf();
            const surfaceScalarField& magSf = mesh.magSf();
            const surfaceScalarField& nodc = mesh.nonOrthDeltaCoeffs();
            const labelUList& own = mesh.owner();
            const labelUList& nei = mesh.neighbour();
            const scalarField& upper = UEqn.upper();
            const scalarField& lower = UEqn.lower();
            const vectorField& Usrc = UEqn.source();

            // ---- momentum diagonal + source, cell loop.
            //      NB: only the RAW lduMatrix diagonal goes in here; the
            //      per-patch internalCoeffs are added in the patch loop
            //      below (aP is the sum of the two and is used for Df only).
            const scalarField& diag = UEqn.diag();

            for (label celli = 0; celli < mesh.nCells(); celli++)
            {
                const label r0 = blockSize*gCell[celli];

                for (label c = 0; c < 3; c++)
                {
                    if (!solveComp[c])
                    {
                        // empty direction: U_c = 0
                        sys.add(r0 + c, r0 + c, 1.0);
                        continue;
                    }

                    sys.add(r0 + c, r0 + c, diag[celli] + dtauDiag[celli]);
                    sys.addRhs(r0 + c, Usrc[celli][c]);

                    if (pseudoTransient)
                    {
                        sys.addRhs(r0 + c, dtauDiag[celli]*U[celli][c]);
                    }
                }
            }

            // ---- momentum off-diagonals, internal faces
            forAll(own, facei)
            {
                const label rO = blockSize*gCell[own[facei]];
                const label rN = blockSize*gCell[nei[facei]];

                for (label c = 0; c < 3; c++)
                {
                    if (!solveComp[c]) continue;

                    sys.add(rO + c, rN + c, upper[facei]);
                    sys.add(rN + c, rO + c, lower[facei]);
                }
            }

            // ---- implicit pressure gradient, internal faces
            //      row(u_P) += Sf.c*(w p_P + (1-w) p_N); row(u_N) -= same
            forAll(own, facei)
            {
                const label rO = blockSize*gCell[own[facei]];
                const label rN = blockSize*gCell[nei[facei]];
                const scalar wf = w[facei];
                const vector& Sff = Sf[facei];

                for (label c = 0; c < 3; c++)
                {
                    if (!solveComp[c]) continue;

                    sys.add(rO + c, rO + 3,  Sff[c]*wf);
                    sys.add(rO + c, rN + 3,  Sff[c]*(1.0 - wf));
                    sys.add(rN + c, rO + 3, -Sff[c]*wf);
                    sys.add(rN + c, rN + 3, -Sff[c]*(1.0 - wf));
                }
            }

            // ---- continuity rows, internal faces
            forAll(own, facei)
            {
                const label rO = blockSize*gCell[own[facei]];
                const label rN = blockSize*gCell[nei[facei]];
                const scalar wf = w[facei];
                const vector& Sff = Sf[facei];

                const scalar Cf = Df[facei]*magSf[facei]*nodc[facei];
                const scalar gl = Df[facei]*(gradPf[facei] & Seff[facei]);

                for (label c = 0; c < 3; c++)
                {
                    if (!solveComp[c]) continue;

                    sys.add(rO + 3, rO + c,  wf*Sff[c]);
                    sys.add(rO + 3, rN + c,  (1.0 - wf)*Sff[c]);
                    sys.add(rN + 3, rO + c, -wf*Sff[c]);
                    sys.add(rN + 3, rN + c, -(1.0 - wf)*Sff[c]);
                }

                sys.add(rO + 3, rO + 3,  Cf);
                sys.add(rO + 3, rN + 3, -Cf);
                sys.add(rN + 3, rO + 3, -Cf);
                sys.add(rN + 3, rN + 3,  Cf);

                sys.addRhs(rO + 3, -gl);
                sys.addRhs(rN + 3,  gl);
            }

            // ---- boundary patches
            forAll(mesh.boundary(), patchi)
            {
                const fvPatch& fvp = mesh.boundary()[patchi];

                if (fvp.size() == 0)
                {
                    continue;
                }

                const labelUList& fc = fvp.faceCells();
                const vectorField& pSf = Sf.boundaryField()[patchi];
                const scalarField& pMagSf = magSf.boundaryField()[patchi];
                const scalarField& pDf = Df.boundaryField()[patchi];
                const scalarField& pw = w.boundaryField()[patchi];
                const vectorField& pSeff = Seff.boundaryField()[patchi];

                const vectorField& ic = UEqn.internalCoeffs()[patchi];
                const vectorField& bc = UEqn.boundaryCoeffs()[patchi];

                const label bStart =
                    fvp.start() - mesh.nInternalFaces();

                if (fvp.coupled())
                {
                    // ---------------- processor / cyclic: like an internal
                    // face whose neighbour lives elsewhere
                    const scalarField& pNodc =
                        nodc.boundaryField()[patchi];
                    const vectorField& pGradPf =
                        gradPf.boundaryField()[patchi];

                    forAll(fc, i)
                    {
                        const label rO = blockSize*gCell[fc[i]];
                        const label rR = blockSize*nbrGCell[bStart + i];
                        const scalar wf = pw[i];
                        const vector& Sff = pSf[i];

                        // momentum: internalCoeffs -> diagonal,
                        //           -boundaryCoeffs -> remote column
                        for (label c = 0; c < 3; c++)
                        {
                            if (!solveComp[c]) continue;

                            sys.add(rO + c, rO + c, ic[i][c]);
                            sys.add(rO + c, rR + c, -bc[i][c]);

                            // implicit pressure gradient
                            sys.add(rO + c, rO + 3, Sff[c]*wf);
                            sys.add(rO + c, rR + 3, Sff[c]*(1.0 - wf));

                            // continuity: velocity part
                            sys.add(rO + 3, rO + c, wf*Sff[c]);
                            sys.add(rO + 3, rR + c, (1.0 - wf)*Sff[c]);
                        }

                        const scalar Cf = pDf[i]*pMagSf[i]*pNodc[i];
                        sys.add(rO + 3, rO + 3,  Cf);
                        sys.add(rO + 3, rR + 3, -Cf);

                        sys.addRhs
                        (
                            rO + 3,
                           -pDf[i]*(pGradPf[i] & pSeff[i])
                        );
                    }
                }
                else
                {
                    // ---------------- genuine boundary
                    const tmp<scalarField> tpIC
                    (
                        p.boundaryField()[patchi].valueInternalCoeffs(pw)
                    );
                    const tmp<scalarField> tpBC
                    (
                        p.boundaryField()[patchi].valueBoundaryCoeffs(pw)
                    );
                    const scalarField& pIC = tpIC();
                    const scalarField& pBC = tpBC();

                    const tmp<vectorField> tuIC
                    (
                        U.boundaryField()[patchi].valueInternalCoeffs(pw)
                    );
                    const tmp<vectorField> tuBC
                    (
                        U.boundaryField()[patchi].valueBoundaryCoeffs(pw)
                    );
                    const vectorField& uIC = tuIC();
                    const vectorField& uBC = tuBC();

                    // Rhie-Chow pressure terms are active only where the
                    // velocity is NOT prescribed (inlets and no-slip walls
                    // carry an exactly known flux)
                    const bool rcActive =
                        !U.boundaryField()[patchi].fixesValue();

                    const tmp<scalarField> tpgIC
                    (
                        p.boundaryField()[patchi].gradientInternalCoeffs()
                    );
                    const tmp<scalarField> tpgBC
                    (
                        p.boundaryField()[patchi].gradientBoundaryCoeffs()
                    );
                    const scalarField& pgIC = tpgIC();
                    const scalarField& pgBC = tpgBC();

                    const bool zeroFlux = zeroFluxPatch(patchi);

                    forAll(fc, i)
                    {
                        const label celli = fc[i];
                        const label rO = blockSize*gCell[celli];
                        const vector& Sff = pSf[i];

                        for (label c = 0; c < 3; c++)
                        {
                            if (!solveComp[c]) continue;

                            // momentum boundary contributions
                            sys.add(rO + c, rO + c, ic[i][c]);
                            sys.addRhs(rO + c, bc[i][c]);

                            // implicit pressure gradient
                            sys.add(rO + c, rO + 3, Sff[c]*pIC[i]);
                            sys.addRhs(rO + c, -Sff[c]*pBC[i]);

                            // continuity: velocity part (identically zero
                            // through impermeable constraint patches)
                            if (!zeroFlux)
                            {
                                sys.add(rO + 3, rO + c, Sff[c]*uIC[i][c]);
                                sys.addRhs(rO + 3, -Sff[c]*uBC[i][c]);
                            }
                        }

                        if (rcActive && !zeroFlux)
                        {
                            // flux_p = -Df |Sf| (pgIC p_P + pgBC)
                            sys.add
                            (
                                rO + 3,
                                rO + 3,
                               -pDf[i]*pMagSf[i]*pgIC[i]
                            );
                            sys.addRhs
                            (
                                rO + 3,
                                pDf[i]*pMagSf[i]*pgBC[i]
                            );

                            // lagged Rhie-Chow term
                            sys.addRhs
                            (
                                rO + 3,
                               -pDf[i]*(gradP[celli] & pSeff[i])
                            );
                        }
                    }
                }
            }
        }

        if (pinPressure)
        {
            sys.setRowIdentity(pinRow, pRefValue);
        }

        sectionTimer("assembly");
        // ================================================== RESIDUALS
        packX(xOld);
        buildRemote();

        FixedList<scalar, 4> res0(0.0);
        blockResiduals(xOld, res0);

        converged =
        (
            iter > 1
         && res0[0] < resControlU
         && res0[1] < resControlU
         && res0[2] < resControlU
         && res0[3] < resControlP
        );

        // Segregated (turbulence) fields, measured at the end of the previous
        // outer iteration.  A negative entry means "never solved" -> not
        // checked, exactly as convergenceControl treats an absent field.
        forAll(extraRes, i)
        {
            if (extraRes[i] >= 0 && extraRes[i] >= extraResTol[i])
            {
                converged = false;
            }
        }

        if (converged)
        {
            for (label c = 0; c < blockSize; c++)
            {
                Info<< lsolver->name() << ":  Solving for " << cmptName[c]
                    << ", Initial residual = " << res0[c]
                    << ", Final residual = " << res0[c]
                    << ", No Iterations 0" << endl;
            }

            Info<< nl << "COUPLED solution converged in " << iter
                << " iterations" << nl << endl;

            runTime.writeNow();

            Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
                << "  ClockTime = " << runTime.elapsedClockTime() << " s"
                << nl << endl;

            break;
        }

        sectionTimer("residuals");
        // ================================ adaptive inner tolerance (E-W)
        // The nonlinear residual driving the forcing term is the SAME
        // normalised block residual the convergence test above uses, reduced
        // to one number by taking the worst block - no second measure is
        // introduced, so the schedule is tied to exactly the quantity the run
        // is judged on.  Nothing here runs, and nothing is pushed to the
        // backend, unless adaptiveTolerance is on.
        if (adaptiveTolerance)
        {
            const scalar ewRes =
                max(max(res0[0], res0[1]), max(res0[2], res0[3]));

            scalar eta = relTolMax;

            if (iter > 1 && ewResPrev > 0 && ewEtaPrev > 0)
            {
                // std::pow, not pow: Foam::pow(dimensionedScalar, ...) makes
                // the unqualified call ambiguous, exactly as Foam::cbrt does
                // in the pseudo-transient block above.
                eta = adaptiveGamma*std::pow(ewRes/ewResPrev, adaptiveAlpha);

                // Standard safeguard against over-loosening: do not let the
                // forcing term drop faster than the previous one squared-ish
                // when that previous term was still large.
                const scalar ewSafe =
                    adaptiveGamma*std::pow(ewEtaPrev, adaptiveAlpha);

                if (ewSafe > 0.1)
                {
                    eta = max(eta, ewSafe);
                }
            }

            eta = min(relTolMax, max(relTolMin, eta));

            lsolver->setRelTol(eta);

            // Separate line, deliberately not in the "Solving for" format, so
            // foamLog and PyFoam residual parsing is untouched.
            Info<< "adaptive inner tolerance: relTol = " << eta
                << ", nonlinear residual = " << ewRes
                << ", previous = " << ewResPrev << endl;

            ewResPrev = ewRes;
            ewEtaPrev = eta;
        }

        sectionTimer("tolerance");
        // ====================================================== SOLVE
        xNew = xOld;
        const label nLinIter = lsolver->solve(sys, xNew);

        forAll(U, celli)
        {
            U[celli] = vector
            (
                xNew[blockSize*celli + 0],
                xNew[blockSize*celli + 1],
                xNew[blockSize*celli + 2]
            );
            p[celli] = xNew[blockSize*celli + 3];
        }
        U.correctBoundaryConditions();
        p.correctBoundaryConditions();

        buildRemote();

        FixedList<scalar, 4> res1(0.0);
        blockResiduals(xNew, res1);

        for (label c = 0; c < blockSize; c++)
        {
            Info<< lsolver->name() << ":  Solving for " << cmptName[c]
                << ", Initial residual = " << res0[c]
                << ", Final residual = " << res1[c]
                << ", No Iterations " << nLinIter << endl;
        }

        sectionTimer("solve+residuals");
        // ================================ conservative Rhie-Chow flux
        // Uses the SAME Df and the SAME lagged gradient as the assembly,
        // so div(phi) == 0 to machine precision in every cell.
        {
            const surfaceScalarField& w = mesh.weights();
            const surfaceVectorField& Sf = mesh.Sf();
            const surfaceScalarField& magSf = mesh.magSf();
            const surfaceScalarField& nodc = mesh.nonOrthDeltaCoeffs();
            const labelUList& own = mesh.owner();
            const labelUList& nei = mesh.neighbour();

            forAll(own, facei)
            {
                const label o = own[facei];
                const label n = nei[facei];
                const scalar wf = w[facei];

                const vector Uf(wf*U[o] + (1.0 - wf)*U[n]);
                const scalar Cf = Df[facei]*magSf[facei]*nodc[facei];

                phi[facei] =
                    (Uf & Sf[facei])
                  - Cf*(p[n] - p[o])
                  + Df[facei]*(gradPf[facei] & Seff[facei]);
            }

            List<vector> nbrU;
            List<scalar> nbrP;
            syncTools::swapBoundaryCellList(mesh, U.primitiveField(), nbrU);
            syncTools::swapBoundaryCellList(mesh, p.primitiveField(), nbrP);

            forAll(mesh.boundary(), patchi)
            {
                const fvPatch& fvp = mesh.boundary()[patchi];

                if (fvp.size() == 0)
                {
                    continue;
                }

                fvsPatchScalarField& pphi = phi.boundaryFieldRef()[patchi];

                const labelUList& fc = fvp.faceCells();
                const vectorField& pSf = Sf.boundaryField()[patchi];
                const scalarField& pMagSf = magSf.boundaryField()[patchi];
                const scalarField& pDf = Df.boundaryField()[patchi];
                const scalarField& pw = w.boundaryField()[patchi];
                const vectorField& pSeff = Seff.boundaryField()[patchi];
                const label bStart =
                    fvp.start() - mesh.nInternalFaces();

                if (fvp.coupled())
                {
                    const scalarField& pNodc = nodc.boundaryField()[patchi];
                    const vectorField& pGradPf =
                        gradPf.boundaryField()[patchi];

                    forAll(fc, i)
                    {
                        const scalar wf = pw[i];
                        const vector Uf
                        (
                            wf*U[fc[i]] + (1.0 - wf)*nbrU[bStart + i]
                        );
                        const scalar Cf = pDf[i]*pMagSf[i]*pNodc[i];

                        pphi[i] =
                            (Uf & pSf[i])
                          - Cf*(nbrP[bStart + i] - p[fc[i]])
                          + pDf[i]*(pGradPf[i] & pSeff[i]);
                    }
                }
                else
                {
                    const tmp<vectorField> tuIC
                    (
                        U.boundaryField()[patchi].valueInternalCoeffs(pw)
                    );
                    const tmp<vectorField> tuBC
                    (
                        U.boundaryField()[patchi].valueBoundaryCoeffs(pw)
                    );
                    const vectorField& uIC = tuIC();
                    const vectorField& uBC = tuBC();

                    const bool rcActive =
                        !U.boundaryField()[patchi].fixesValue();

                    const tmp<scalarField> tpgIC
                    (
                        p.boundaryField()[patchi].gradientInternalCoeffs()
                    );
                    const tmp<scalarField> tpgBC
                    (
                        p.boundaryField()[patchi].gradientBoundaryCoeffs()
                    );
                    const scalarField& pgIC = tpgIC();
                    const scalarField& pgBC = tpgBC();

                    const bool zeroFlux = zeroFluxPatch(patchi);

                    forAll(fc, i)
                    {
                        const label celli = fc[i];

                        if (zeroFlux)
                        {
                            pphi[i] = 0.0;
                            continue;
                        }

                        scalar f = 0.0;

                        for (label c = 0; c < 3; c++)
                        {
                            f += pSf[i][c]*(uIC[i][c]*U[celli][c] + uBC[i][c]);
                        }

                        if (rcActive)
                        {
                            f -=
                                pDf[i]*pMagSf[i]
                               *(pgIC[i]*p[celli] + pgBC[i]);
                            f += pDf[i]*(gradP[celli] & pSeff[i]);
                        }

                        pphi[i] = f;
                    }
                }
            }
        }

        sectionTimer("flux");
        // ======================================== continuity reporting
        {
            scalarField divPhi(mesh.nCells(), 0.0);
            const labelUList& own = mesh.owner();
            const labelUList& nei = mesh.neighbour();

            forAll(own, facei)
            {
                divPhi[own[facei]] += phi[facei];
                divPhi[nei[facei]] -= phi[facei];
            }

            forAll(mesh.boundary(), patchi)
            {
                const labelUList& fc = mesh.boundary()[patchi].faceCells();
                const fvsPatchScalarField& pphi =
                    phi.boundaryField()[patchi];

                forAll(fc, i)
                {
                    divPhi[fc[i]] += pphi[i];
                }
            }

            scalar sumLocal = 0.0;
            scalar sumLocalNoPin = 0.0;
            scalar sumGlobal = 0.0;
            scalar maxLocal = 0.0;

            for (label celli = 0; celli < mesh.nCells(); celli++)
            {
                sumLocal += mag(divPhi[celli]);
                sumGlobal += divPhi[celli];
                maxLocal = max(maxLocal, mag(divPhi[celli]));

                if (!(pinPressure && gCell[celli] == pRefCell))
                {
                    sumLocalNoPin += mag(divPhi[celli]);
                }
            }

            reduce(sumLocal, sumOp<scalar>());
            reduce(sumLocalNoPin, sumOp<scalar>());
            reduce(sumGlobal, sumOp<scalar>());
            reduce(maxLocal, maxOp<scalar>());

            scalar refFlux = 0.0;
            forAll(phi.primitiveField(), facei)
            {
                refFlux += mag(phi[facei]);
            }
            forAll(mesh.boundary(), patchi)
            {
                const fvsPatchScalarField& pphi =
                    phi.boundaryField()[patchi];
                forAll(pphi, i)
                {
                    refFlux += mag(pphi[i]);
                }
            }
            reduce(refFlux, sumOp<scalar>());
            refFlux = max(refFlux, vSmall);

            cumulativeContErr += sumGlobal;

            Info<< "time step continuity errors : sum local = " << sumLocal
                << ", global = " << sumGlobal
                << ", cumulative = " << cumulativeContErr << endl;

            Info<< "coupled assembly check: sum|div(phi)| = " << sumLocal
                << " (excl. pRef cell " << sumLocalNoPin << ")"
                << ", max|div(phi)| = " << maxLocal
                << ", normalised = " << sumLocalNoPin/refFlux << endl;
        }

        sectionTimer("continuity");
        // ================================ segregated turbulence update
        // phi has just been rebuilt from the coupled solution, so the
        // transport equations see the same conservative flux the coupled
        // system produced - the turbulence model must never run on a stale
        // flux.  correct() solves k/epsilon/omega/... with the fvSolution
        // "solvers" and "relaxationFactors" entries and prints the standard
        // "Solving for k, Initial residual = ..." lines, so the log stays
        // foamLog/PyFoam parseable.  For a laminar (Stokes) model both calls
        // are no-ops, which is why the laminar path is bit-for-bit unchanged.
        viscosity->correct();
        momentumTransport->correct();

        sectionTimer("turbulence");
        // ================================== segregated energy transport
        // Solved on the SAME conservative Rhie-Chow flux the coupled system
        // produced, and after the turbulence update so that nut is the one
        // just corrected on that flux.  The equation, the relaxation lookup
        // and the residual bookkeeping are the ones a segregated solver uses,
        // so `foamRun -solver incompressibleFluid` plus the stock
        // scalarTransport functionObject solves the identical discrete
        // equation - which is what doc/VALIDATION.md 7 measures.
        //
        // T is a PASSIVE scalar: constant properties, no buoyancy, no
        // viscous heating, so nothing computed here re-enters the momentum
        // rows and the coupled assembly is untouched.  doc/USAGE.md 9 states
        // the limits that follow.
        if (solveEnergy)
        {
            volScalarField& T = TPtr();

            // Simple gradient-diffusion hypothesis.  Identical to the D of
            // the scalarTransport functionObject with alphal = 1/Pr and
            // alphat = 1/Prt, which is what makes that function object an
            // exact segregated reference.  nut() is a zero field for a
            // laminar model, leaving alphaEff = nu/Pr.
            const volScalarField alphaEff
            (
                "alphaEff",
                momentumTransport->nu()/Pr + momentumTransport->nut()/Prt
            );

            fvScalarMatrix TEqn
            (
                fvm::div(phi, T)
              - fvm::laplacian(alphaEff, T)
            );

            // No-op unless relaxationFactors/equations carries a T entry:
            // with no entry fvMatrix::relaxationFactor() returns 0 and
            // relax() returns immediately.  Same mechanism, same key and
            // same behaviour as the turbulence equations, rather than a
            // private relaxation control.
            if (conjugate.valid())
            {
                // Fluid and solids in one system; same fluid equation.
                conjugate->solve(phi, alphaEff);
            }
            else
            {
                TEqn.relax();

                TEqn.solve();
            }

            // Discrete energy balance, the temperature counterpart of the
            // continuity check above and a permanent verification print.
            // Summing the assembled equation over every cell telescopes the
            // internal faces away and leaves exactly the boundary fluxes, so
            // on a converged solution the net convective outflow of T must
            // equal the net diffusive influx.  Both sums are taken with the
            // boundary conditions' own coefficient functions, so they are the
            // fluxes the matrix used and not a reconstruction of them (the
            // explicit non-orthogonal correction of the laplacian is the one
            // term outside this identity, exactly as for the pressure).
            scalar hOut = 0.0;      // sum_b phi_b T_b       [K m^3/s]
            scalar qBoundary = 0.0; // sum_b alphaEff |Sf| snGrad(T)
            scalar qWall = 0.0;     // the same, wall patches only

            forAll(mesh.boundary(), patchi)
            {
                const fvPatch& fvp = mesh.boundary()[patchi];

                if (fvp.size() == 0 || fvp.coupled())
                {
                    continue;
                }

                const fvPatchScalarField& pT = T.boundaryField()[patchi];
                const scalarField& pAlpha =
                    alphaEff.boundaryField()[patchi];
                const scalarField& pMagSf =
                    mesh.magSf().boundaryField()[patchi];
                const fvsPatchScalarField& pphi =
                    phi.boundaryField()[patchi];
                const labelUList& fc = fvp.faceCells();

                const tmp<scalarField> tgIC(pT.gradientInternalCoeffs());
                const tmp<scalarField> tgBC(pT.gradientBoundaryCoeffs());
                const scalarField& gIC = tgIC();
                const scalarField& gBC = tgBC();

                const bool isWall = isA<wallFvPatch>(fvp);

                forAll(fc, i)
                {
                    const scalar q =
                        pAlpha[i]*pMagSf[i]*(gIC[i]*T[fc[i]] + gBC[i]);

                    hOut += pphi[i]*pT[i];
                    qBoundary += q;

                    if (isWall)
                    {
                        qWall += q;
                    }
                }
            }

            reduce(hOut, sumOp<scalar>());
            reduce(qBoundary, sumOp<scalar>());
            reduce(qWall, sumOp<scalar>());

            const scalar imbalance = hOut - qBoundary;
            const scalar qScale =
                max(max(mag(hOut), mag(qBoundary)), vSmall);

            Info<< "coupled energy check: wall heat flux = " << qWall
                << ", enthalpy outflow = " << hOut
                << ", boundary diffusion = " << qBoundary
                << ", imbalance = " << 100.0*imbalance/qScale << " %"
                << endl;
        }

        // Read back the initial residuals of the segregated fields for the
        // next outer iteration's convergence test
        if (extraResName.size())
        {
            const DynamicList<word> resFields
            (
                convergenceControl::getFieldNames(mesh)
            );

            forAll(extraResName, i)
            {
                bool haveField = false;
                forAll(resFields, j)
                {
                    if (resFields[j] == extraResName[i])
                    {
                        haveField = true;
                        break;
                    }
                }

                if (!haveField)
                {
                    continue;
                }

                scalar r0 = -1.0;
                scalar r = -1.0;
                convergenceControl::getInitialResiduals
                (
                    mesh,
                    extraResName[i],
                    0,
                    r0,
                    r
                );
                extraRes[i] = r0;
            }
        }

        runTime.write();

        Info<< "ExecutionTime = " << runTime.elapsedCpuTime() << " s"
            << "  ClockTime = " << runTime.elapsedClockTime() << " s"
            << nl << endl;
    }

    if (!converged)
    {
        Info<< nl << "COUPLED did NOT converge within " << nOuterIterations
            << " outer iterations" << nl << endl;
        runTime.writeNow();
    }

    Info<< "End\n" << endl;

    #ifdef HAVE_AMGX
    // Drop the start-up reference; the last owner of AmgX handles (destroyed
    // as main's objects go out of scope) finalizes the library.
    if (amgxEarly)
    {
        Foam::amgxCommon::release();
    }
    #endif

    return 0;
}


// ************************************************************************* //
