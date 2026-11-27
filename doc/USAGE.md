# Using coupledSimpleFoam

This is the operating manual: how to run an existing steady incompressible
OpenFOAM case with `coupledSimpleFoam` instead of the segregated solver, what
every dictionary key does, how to choose a linear solver and preconditioner,
and what to do when a run stalls.

`coupledSimpleFoam` is a steady-state incompressible solver for OpenFOAM 14
(openfoam.org) that assembles momentum and continuity as a **single implicitly
coupled linear system**. The unknowns are interleaved per cell as `[u, v, w, p]`
with a block size of 4, so the system has `4*nCells` rows. There are no
under-relaxation factors for pressure and velocity; the outer loop is a Picard
iteration on the nonlinear convection term and on the lagged Rhie-Chow
pressure-gradient correction. Turbulence, when present, is handled by
OpenFOAM's own `momentumTransport` framework, solved segregated and lagged one
outer iteration behind the coupled block.

Related documents:

- [doc/THEORY.md](THEORY.md) - the formulation itself: unknown ordering, the
  implicit pressure-gradient block, the Rhie-Chow continuity row, boundary
  coefficients, residual normalisation. Read it for *why* a key exists; read
  this document for *how* to set it.
- [doc/INSTALL.md](INSTALL.md) - prerequisites, build switches, optional PETSc
  builds, and a build/run troubleshooting table
- [doc/DEVELOPMENT.md](DEVELOPMENT.md) - code layout and OpenFOAM API notes
- `doc/BENCHMARKS.md` - the validation cases and performance measurements this
  document quotes

---

## Contents

1. [Case setup checklist](#1-case-setup-checklist)
2. [COUPLED dictionary reference](#2-coupled-dictionary-reference)
3. [Choosing a linear solver and preconditioner](#3-choosing-a-linear-solver-and-preconditioner)
4. [Convergence tuning](#4-convergence-tuning)
5. [Running in parallel](#5-running-in-parallel)
6. [Reading the log](#6-reading-the-log)
7. [Post-processing](#7-post-processing)
8. [Comparing against the segregated solver](#8-comparing-against-the-segregated-solver)
9. [Heat transfer](#9-heat-transfer)
10. [Troubleshooting index](#10-troubleshooting-index)

---

## 1. Case setup checklist

A steady incompressible case that already runs with the segregated solver
(`foamRun -solver incompressibleFluid`) needs **four** things to also run with
`coupledSimpleFoam`. Nothing has to be removed: the extra `COUPLED`
sub-dictionary is ignored by `foamRun`, and the `SIMPLE` and `relaxationFactors`
entries are ignored by `coupledSimpleFoam`, so **one case directory can drive
both solvers**. That property is what makes a fair head-to-head comparison
possible (section 8).

| # | Requirement | Where |
| --- | --- | --- |
| 1 | `viscosityModel` entry | `constant/physicalProperties` |
| 2 | `simulationType` entry - **mandatory even for laminar** | `constant/momentumTransport` |
| 3 | `ddtSchemes { default steadyState; }` and a `div(phi,U)` scheme | `system/fvSchemes` |
| 4 | A `COUPLED` sub-dictionary (optional - every key has a default) | `system/fvSolution` |

Then run the solver in the case directory:

```sh
coupledSimpleFoam
```

There is no `-solver` argument: `coupledSimpleFoam` is a standalone
application, not a `foamRun` solver module.

### 1.1 `constant/`

**`constant/physicalProperties`** - read through
`physicalProperties::findModelDict`, which probes `physicalProperties`, then
`thermophysicalProperties`, then `transportProperties`, so a legacy case with
only `constant/transportProperties` is still found. It must supply a
`viscosityModel` keyword:

```c++
viscosityModel  constant;
nu              1e-05;
```

**`constant/momentumTransport`** - **required in every case, including laminar
ones.** The model dictionary is read by `momentumTransportModel::readModelDict`,
which falls back to `constant/turbulenceProperties`; if neither file exists the
run dies in the `IOdictionary` read before the first iteration. Laminar cases
need only:

```c++
simulationType  laminar;
```

A RANS case selects a model in the usual way:

```c++
simulationType  RAS;

RAS
{
    model           kEpsilon;      // or kOmegaSST, ...
    turbulence      on;
    printCoeffs     on;
}
```

Any incompressible laminar / RAS / LES model whose run-time selection table is
in the linked libraries is selectable, because the solver constructs
`incompressible::momentumTransportModel::New(U, phi, viscosity)` exactly as the
stock solver module does. Note the scope caveat: **only `laminar` / `Stokes`,
`kEpsilon` and `kOmegaSST` have actually been exercised** (see
`doc/BENCHMARKS.md`); other models are available but untested with this solver.

**`constant/polyMesh`** - a normal OpenFOAM mesh. `blockMesh` and
`snappyHexMesh` meshes are both used in the validation set. Non-orthogonality
is handled by the same explicit correction a segregated `pEqn` uses (the
implicit pressure Laplacian uses `nonOrthDeltaCoeffs`, the remainder is lagged
into the right-hand side), and the validation set includes a `snappyHexMesh`
mesh at 80.9 degrees maximum non-orthogonality.

### 1.2 `0/` fields

| Field | Laminar | RANS (`kEpsilon`) | RANS (`kOmegaSST`) |
| --- | --- | --- | --- |
| `p` (kinematic pressure) | required | required | required |
| `U` | required | required | required |
| `phi` | optional | optional | optional |
| `nut` | - | required | required |
| `k` | - | required | required |
| `epsilon` | - | required | - |
| `omega` | - | - | required |
| `T` | only with `solveEnergy yes` | only with `solveEnergy yes` | only with `solveEnergy yes` |

`p` and `U` are `MUST_READ`. `phi` is read if present and computed from `U`
otherwise, so a cold start does not need it; a restart directory written by
either solver will contain it. The turbulence fields and their wall functions
belong to the momentum-transport model, so they are exactly the fields the
segregated solver needs for the same model - stock tutorial `0/` directories
work unchanged.

`T` is **constructed only when `COUPLED/solveEnergy` is on**, so a case that
does not ask for heat transfer needs no `0/T` and is unaffected by the feature
existing. With the switch on and no readable `0/T`, the run stops at start-up
with an error that names `solveEnergy` and lists the wall conditions to choose
from, rather than failing somewhere inside a field read. Section 9 covers the
thermal boundary conditions.

**Boundary conditions.** Every boundary condition that OpenFOAM's own
`fvc::grad` and `fvMatrix` machinery handles is supported, because the assembly
uses `valueInternalCoeffs` / `valueBoundaryCoeffs` for the velocity and
pressure values and `gradientInternalCoeffs` / `gradientBoundaryCoeffs` for the
pressure gradient rather than special-casing individual types. Points worth
knowing:

- **Pressure level.** If any `p` patch reports `fixesValue()` - a `fixedValue`
  outlet, say - that patch sets the level and nothing is pinned. Otherwise the
  continuity row of `pRefCell` is replaced by the identity row
  `p[pRefCell] = pRefValue`. The mode is echoed at start-up.
- **Rhie-Chow at prescribed-velocity patches.** The Rhie-Chow pressure terms
  are switched off on any patch where `U` reports `fixesValue()`, because
  inlets and no-slip walls carry an exactly known flux. Walls therefore
  contribute nothing to the continuity row and inlets contribute `U_b . Sf` to
  the right-hand side.
- **Impermeable constraint patches.** `symmetry`, `symmetryPlane`, `wedge` and
  `slip`-`U` patches are forced to exactly zero mass flux, both in the
  continuity row and in the reconstructed `phi`. Their per-component transform
  coefficients cannot represent zero flux on tilted faces, and treating them
  generically leaks mass: on the 3-D interrupted-fin unit cell that leak was
  worth 9.6 % in pressure drop before the rule was introduced.
- **`empty` patches** need no special handling: they have zero size, so patch
  loops skip them. Empty *directions* get an identity momentum row (`U_c = 0`),
  matching what `fvMatrix::solveSegregated` does for invalid components.
- **`processor` patches** are assembled as internal faces whose neighbour lives
  elsewhere, using the same coefficients OpenFOAM uses for its own interface
  updates. This path is exercised by the serial-versus-parallel agreement
  reported in [doc/VALIDATION.md](VALIDATION.md).
- **`cyclic` patches** take that same coupled branch and are therefore
  structurally supported, but **no shipped case exercises them**: treat a
  periodic case as unverified and check global mass balance yourself.
- **`partialSlip` is not covered by the zero-flux rule.** Only `symmetry`,
  `symmetryPlane`, `wedge` and full `slip` patches are forced to exactly zero
  flux; a `partialSlip` wall will leak mass through the patch.
- **`wedge` (axisymmetric) patches are untested.** They are at least
  zero-flux-correct in the continuity row, but the component-mixing transform
  is not accounted for in the pressure-gradient block.
- **Non-conformal / stitched (`nonConformalCoupled`) patches are not handled.**
  They take the generic coupled branch, which does not implement their
  interpolation, so a case using them runs but is silently wrong.

**Mass balance.** There is no `adjustPhi`-style rescaling. A fixed-flux domain
whose inflow and outflow do not balance produces a singular continuity block;
give at least one outlet a `fixedValue` pressure or a
`zeroGradient`/`inletOutlet` velocity as usual.

### 1.3 `system/fvSchemes`

```c++
ddtSchemes
{
    default         steadyState;        // MANDATORY - there is no ddt term
}

gradSchemes
{
    default         Gauss linear;
}

divSchemes
{
    default         none;
    div(phi,U)      bounded Gauss upwind;

    // RANS only - one entry per transported turbulence field
    div(phi,k)      bounded Gauss upwind;
    div(phi,epsilon) bounded Gauss upwind;
    // div(phi,omega) bounded Gauss upwind;   // kOmegaSST

    // solveEnergy only
    div(phi,T)      Gauss linearUpwind grad(T);
}

laplacianSchemes
{
    default         Gauss linear corrected;
}

interpolationSchemes
{
    default         linear;
}

snGradSchemes
{
    default         corrected;
}

// kOmegaSST only
wallDist
{
    method          meshWave;
}
```

Notes:

- **`steadyState` is mandatory.** The solver builds no `fvm::ddt` term at all;
  a transient `ddtScheme` would simply have nothing to act on. Pseudo-transient
  continuation (section 4) adds a diagonal term directly and does not go
  through `ddtSchemes`.
- **`div(devTauCorrFlux)` needs no entry.** The explicit `dev2(T(grad U))`
  stress flux enters through `fvm::divc`, which is `fvc::surfaceIntegrate` of a
  surface field, not a Gauss scheme. Legacy `div(devTauCorrFlux)` or
  `div((nuEff*dev2(T(grad(U)))))` lines in an existing `fvSchemes` are
  vestigial and harmless.
- **`wallDist` is required by `kOmegaSST`** and by any other model that asks
  for a wall distance. Without it the run fails in the model's construction.
- **`solveEnergy` needs `div(phi,T)` and `laplacian(alphaEff,T)`.** The
  diffusivity field the solver builds is named `alphaEff`, so that is the name
  in the laplacian scheme lookup; a `laplacianSchemes { default ... }` entry
  covers it, which is what every case here has. The stock `scalarTransport`
  function object used as the segregated reference names its own diffusivity
  `DT` and therefore looks up `laplacian(DT,T)` - another reason to leave the
  laplacian `default` in place rather than enumerating every name.
- Convection scheme choice is a normal steady-state decision.
  `bounded Gauss upwind` is the robust default and is what the validation set
  uses; `bounded Gauss linearUpwind grad(U)` is the usual second-order step up
  once a first-order solution exists (section 4.5).

### 1.4 `system/fvSolution`

Keep the `solvers` sub-dictionary: the coupled p-U block does not use it, but
the **segregated transport equations do** - the turbulence fields are solved by
`momentumTransport->correct()` through OpenFOAM's normal machinery, and `T` by
the energy step, so `k`, `epsilon`, `omega` and `T` need `solvers` entries and
honour `relaxationFactors` exactly as in a segregated run. The stock tutorial
entries are fine.

`SIMPLE` and the `relaxationFactors` entries for `p` and `U` are **read by
nobody** in a coupled run - the momentum matrix is never relaxed and there is
no pressure-correction field to relax. Leaving them in place is harmless and is
what makes one case directory serve both solvers.

A complete minimal `fvSolution` for a laminar case:

```c++
solvers
{
    p
    {
        solver          GAMG;
        tolerance       1e-10;
        relTol          0.01;
        smoother        GaussSeidel;
    }

    "(U|U.*)"
    {
        solver          smoothSolver;
        smoother        symGaussSeidel;
        tolerance       1e-10;
        relTol          0.01;
    }
}

// ---- segregated reference: foamRun -solver incompressibleFluid ----------
SIMPLE
{
    nNonOrthogonalCorrectors 0;

    pRefCell        0;
    pRefValue       0;

    residualControl
    {
        p               1e-4;
        U               1e-5;
    }
}

relaxationFactors
{
    equations { U 0.7; }
    fields    { p 0.3; }
}

// ---- fully coupled solver: coupledSimpleFoam ---------------------------
COUPLED
{
    nOuterIterations    100;

    linearSolver        SparseLU;   // serial direct; PETSc for MPI / 3-D

    pRefCell            0;
    pRefValue           0;

    residualControl
    {
        U               1e-5;
        p               1e-4;
    }
}
```

### 1.5 `system/controlDict`

`controlDict` is used for time bookkeeping, writing and function objects, but
**it does not control the outer loop**: the loop runs until `nOuterIterations`
or convergence, whichever comes first, incrementing the time index by one per
outer iteration. Practical consequences:

- `endTime` and `stopAt` do not stop the run. Cap iterations with
  `nOuterIterations`.
- `writeControl` / `writeInterval` govern intermediate writes exactly as usual;
  with `writeControl timeStep` and a `writeInterval` larger than the expected
  iteration count, no intermediate directory is written.
- The final state is always written, whether the run converged or hit its
  iteration cap - the solver calls `writeNow()` in both cases.
- `purgeWrite` behaves normally, so a small value keeps the case directory
  clean during a long run.
- Set `writePrecision 12` when the results will be compared numerically
  against another run; the ASCII precision otherwise becomes the noise floor of
  the comparison.
- The `solver incompressibleFluid;` entry that `foamRun` needs is ignored by
  `coupledSimpleFoam`, so it can stay.
- **`functions { ... }` entries are not executed.** OpenFOAM triggers function
  objects from `Time::run()`, and this solver's outer loop is bounded by
  `nOuterIterations` rather than by the time loop, so it never calls it. Run
  function objects afterwards on the written time directories with
  `foamPostProcess` (section 7). Leaving a `functions` block in `controlDict`
  is harmless - it is simply inert during the run, and is picked up normally
  when the same case is run with `foamRun`.

### 1.6 Out of scope

The solver assembles the momentum and continuity equations only. Not
supported, and silently absent rather than diagnosed:

- `fvModels` / `fvConstraints` (so no `fvOptions`-style sources, no porous
  zones, no explicit turbulence sources)
- MRF, mesh motion, `adjustPhi`
- energy equation, conjugate heat transfer, multiphase, compressibility
- transient time-accurate simulation

The former `divDevTauCorrection` switch no longer exists; the momentum operator
always carries the explicit `dev2(T(grad U))` flux, supplied by OpenFOAM
itself. A stale `divDevTauCorrection yes;` entry in a case dictionary is
ignored.

---

## 2. `COUPLED` dictionary reference

The `COUPLED` sub-dictionary lives in `system/fvSolution` alongside `solvers`,
`SIMPLE` and `relaxationFactors`. **If it is absent entirely, every default
below applies** - the solver runs with 100 outer iterations, the serial direct
backend (or PETSc in parallel), `residualControl U 1e-5 / p 1e-4`, and the
pressure pinned at global cell 0.

### 2.1 Annotated dictionary

> The annotated dictionary below shows `tolerance` and `maxLinearIter` twice - once with the Eigen default and once with the PETSc default - to document both backends. OpenFOAM accepts duplicate keys silently and the LAST one wins, so delete the pair that does not apply before copying this dictionary into a case.

```c++
COUPLED
{
    // ---- outer loop -----------------------------------------------------
    nOuterIterations    100;      // outer-iteration cap

    // ---- backend --------------------------------------------------------
    linearSolver        SparseLU; // SparseLU | BiCGSTAB (Eigen, serial only)
                                  // PETSc (serial and MPI)

    // ---- pressure level --------------------------------------------------
    pRefCell            0;        // GLOBAL cell index
    pRefValue           0;        // used only when no fixedValue-p patch exists

    // ---- discretisation --------------------------------------------------
    rhieChowFactor      1.0;      // scale on D_f = V/aP

    // ---- pseudo-transient continuation -----------------------------------
    pseudoTransient     no;
    pseudoCo            200;
    pseudoUref          0;        // 0 -> auto, from max|U|

    // ---- Eigen backend ---------------------------------------------------
    tolerance           1e-10;    // BiCGSTAB tolerance
    maxLinearIter       2000;
    dropTolerance       1e-5;     // BiCGSTAB + IncompleteLUT only
    fillFactor          20;       // BiCGSTAB + IncompleteLUT only

    // ---- PETSc backend ---------------------------------------------------
    tolerance           1e-30;    // absolute KSP tolerance
    relTol              1e-12;    // relative KSP tolerance
    maxLinearIter       5000;
    reusePreconditioner no;       // freeze the PC after the first build
    pcReuseInterval     1;        // 1 = rebuild the PC every outer iteration
    petscOptions        "...";    // see 2.4 for the shipped default

    // ---- adaptive inner tolerance (Eisenstat-Walker) ---------------------
    adaptiveTolerance   no;       // no -> relTol above is used unchanged
    relTolMax           1e-1;     // loosest inner relative tolerance
    relTolMin           1e-8;     // tightest inner relative tolerance, and
                                  // the floor
    adaptiveGamma       0.9;      // forcing-term factor
    adaptiveAlpha       1.5;      // forcing-term exponent (superlinear for
                                  // 1 < alpha <= 2)

    // ---- heat transfer ---------------------------------------------------
    solveEnergy         no;       // yes -> solve T; requires a 0/T field
    Pr                  0.71;     // laminar Prandtl number
    Prt                 0.85;     // turbulent Prandtl number (RANS only)

    // ---- convergence -----------------------------------------------------
    residualControl
    {
        U               1e-5;     // coupled momentum blocks
        p               1e-4;     // coupled continuity block
        k               1e-4;     // any other key = segregated field
        epsilon         1e-4;     // exact field names only, no regexes
        T               1e-6;     // checked only when solveEnergy is on
    }
}
```

`tolerance` and `maxLinearIter` are shared key names whose **defaults differ by
backend**; only the active backend reads them.

### 2.2 Every key

| Key | Type | Default | Meaning and when to change it |
| --- | --- | --- | --- |
| `nOuterIterations` | `label` | `100` | Cap on outer (Picard) iterations. Laminar cases converge in 12-30, from a 4,096-cell cavity to a 2.15M-cell duct alike; RANS cases need several hundred, so raise it to ~1000 for RANS. The run writes its final state and reports `COUPLED did NOT converge` when the cap is hit. |
| `linearSolver` | `word` | `PETSc` in a parallel run, `SparseLU` in serial | Backend selection. `SparseLU` and `BiCGSTAB` are Eigen and **serial only** - requesting either in a parallel run is a fatal error. Requesting `PETSc` in a binary built without PETSc is a fatal error naming the fix. See section 3. |
| `pRefCell` | `label` | `0` | **Global** cell index (cell 0 lives on rank 0) whose continuity row becomes the identity row when the pressure level is not fixed by a boundary. Change it only to avoid pinning inside a degenerate cell. |
| `pRefValue` | `scalar` | `0` | Value pinned at `pRefCell`. Used only when no `p` patch reports `fixesValue()`. |
| `rhieChowFactor` | `scalar` | `1.0` | Multiplier on the Rhie-Chow coefficient `D_f = V/aP`. `1.0` is the natural coupled choice, since a coupled solve has no relaxation factor. Set it to a segregated reference run's momentum relaxation factor **only** when reproducing that reference exactly for verification - see section 8.3. |
| `pseudoTransient` | `Switch` | `no` | Enable pseudo-transient continuation: adds `V/dtau` to the three momentum diagonals (and hence to `aP`, and hence to `D_f`). A relaxation device, not a time-accurate scheme. See section 4.3. |
| `pseudoCo` | `scalar` | `200` | Target pseudo-Courant number setting the local step `dtau = pseudoCo*h/max(mag(U), 0.05*Uref)` with `h = cbrt(V)`. Lower is more damping. Has no effect unless `pseudoTransient` is on. |
| `pseudoUref` | `scalar` | `0` | Reference velocity for the `0.05*Uref` floor. `0` means auto: the global maximum of `mag(U)` over internal cells and all boundary faces, recomputed each outer iteration. Set it explicitly when the starting field is zero everywhere except a small inlet, so that the floor is meaningful from the first iteration. |
| `tolerance` | `scalar` | Eigen `1e-10`, PETSc `1e-30` | Absolute linear-solver tolerance. Eigen: the `BiCGSTAB` iterative tolerance (unused by `SparseLU`, which is direct). PETSc: the `KSPSetTolerances` absolute tolerance, deliberately set unreachably tight so that `relTol` is what stops the inner solve. |
| `maxLinearIter` | `label` | Eigen `2000`, PETSc `5000` | Inner iteration cap. Unused by `SparseLU`. For large RANS systems, capping this at ~500 and pairing it with a looser `relTol` is the measured recipe (section 4.4). |
| `dropTolerance` | `scalar` | `1e-4` | Eigen `IncompleteLUT` drop tolerance. `BiCGSTAB` path only. (Was `1e-5`; with `fillFactor 5` the preconditioner builds about 10x faster and the outer loop converges in the same number of iterations.) |
| `fillFactor` | `label` | `5` | Eigen `IncompleteLUT` fill factor. `BiCGSTAB` path only. (Was `20`.) |
| `relTol` | `scalar` | `1e-12` | PETSc relative KSP tolerance - **the tolerance that actually terminates the inner solve**. Tight by default because the outer loop is a Picard iteration started from a nonzero initial guess: a loose inner solve makes the linear update negligible and the outer loop plateaus. See section 4.2 for the regime table. Ignored when `adaptiveTolerance` is on, which computes the value per outer iteration instead (section 4.7). |
| `adaptiveTolerance` | `Switch` | `no` | Compute the inner relative tolerance per outer iteration from an Eisenstat-Walker forcing term instead of holding `relTol` fixed - loose while the outer residual is large, tightening as it falls. **Off by default, and with it off nothing about the run changes**: no tolerance is pushed to the backend and no extra line is printed. The adaptive path also measures the inner residual against the initial residual of that solve rather than against `\|\|b\|\|`, so its numbers are not interchangeable with `relTol`'s. PETSc, AmgX and Eigen `BiCGSTAB` (measured against that solve's initial residual); `SparseLU` is direct and ignores it (section 4.7). |
| `relTolMax` | `scalar` | `1e-1` | Loosest inner relative tolerance the schedule may ask for, and the value used on the first outer iteration. Only read when `adaptiveTolerance` is on. |
| `relTolMin` | `scalar` | `1e-8` | Tightest inner relative tolerance, i.e. the floor the schedule clamps to. Only read when `adaptiveTolerance` is on. Lower it towards `1e-12` if the outer loop stalls at a level the fixed-tolerance run passed through. A value above `relTolMax` is a fatal error at start-up rather than a silent clamp. |
| `adaptiveGamma` | `scalar` | `0.9` | Eisenstat-Walker forcing parameter, the multiplier on the residual ratio. Lower means a tighter schedule throughout. Only read when `adaptiveTolerance` is on. |
| `adaptiveAlpha` | `scalar` | `1.5` | Eisenstat-Walker exponent on the residual ratio. `1 < alpha <= 2` gives the superlinear tightening this is chosen for; `2` is the classic quadratic choice and tightens hardest. Only read when `adaptiveTolerance` is on. |
| `reusePreconditioner` | `bool` | `no` | Freeze the PETSc preconditioner after its first build and keep it for the whole run (`KSPSetReusePreconditioner`). **Measured counterproductive** - see section 4.6. Present for experimentation, not for production. Takes precedence over `pcReuseInterval`, which is then ignored with a warning. |
| `pcReuseInterval` | `label` | `1` | Rebuild the PETSc preconditioner on outer iterations `1, N+1, 2N+1, ...` and reuse it in between. `1` (the default) rebuilds every outer iteration, which is the historical behaviour and leaves the run unchanged. Unlike `reusePreconditioner` this is a *periodic refresh*, not a permanent freeze - the reuse flag is explicitly turned back off on a rebuild iteration. Must be `>= 1`; anything less is a fatal error. See section 4.7. |
| `petscOptions` | `string` | the fieldsplit-Schur + hypre string in 2.4 | Free-form PETSc options string, inserted with `PetscOptionsInsertString` after `PetscInitialize`. Everything about the preconditioner and the Krylov method is set here; no rebuild is ever needed to change it. Also settable through the `PETSC_OPTIONS` environment variable. |
| `solveEnergy` | `Switch` | `no` | Solve a temperature transport equation, segregated, after the coupled p-U solve of each outer iteration. **With it off nothing about the run changes**: `T` is not constructed, no `0/T` is read and no extra line is printed. With it on the case must provide `0/T`, a `solvers { T { ... } }` entry and `div(phi,T)` / `laplacian(alphaEff,T)` schemes; a missing `0/T` is a fatal error naming the switch. `T` is a **passive** scalar and never feeds back into momentum - see section 9. |
| `Pr` | `scalar` | `0.71` | Laminar Prandtl number, used as `alphaEff = nu/Pr + nut/Prt`. Only read when `solveEnergy` is on. Must be positive. Air is 0.71, water at 20 C about 7. It lives here rather than in `constant/physicalProperties` because for a passive scalar it is a coefficient of the transport equation, not a property the viscosity model knows about - the same choice OpenFOAM's own `scalarTransport` function object makes with its `alphal`. |
| `Prt` | `scalar` | `0.85` | Turbulent Prandtl number. Only read when `solveEnergy` is on, and only has an effect when a RAS model makes `nut` non-zero. Must be positive. |
| `residualControl` | `dictionary` | see below | Convergence thresholds. |
| `residualFloor` | `scalar` | `1e-3` | Floor on each velocity component's residual normalisation, as a fraction of the largest velocity component's. A component that carries next to none of the flow (the cross-stream velocity of a periodic or symmetric case) otherwise has a normalisation made of round-off, so its residual stays O(1) and `residualControl` never fires although the flow has converged. `0` restores plain OpenFOAM normalisation. |

### 2.3 `residualControl`

| Key | Default | Meaning |
| --- | --- | --- |
| `U` | `1e-5` | Threshold on all three coupled momentum blocks (`Ux`, `Uy`, `Uz` each tested separately against this one value). |
| `p` | `1e-4` | Threshold on the coupled continuity block. |
| `T` | none | Only meaningful with `solveEnergy yes`, and then handled by the same "any other key" rule below: `T` is a segregated field like `k` or `epsilon`. With `solveEnergy no` the entry is inert, because `T` never reaches the residual store. |
| *any other key* | none - the value given is used as-is | Treated as a **segregated** field. Its initial residual is read back from OpenFOAM's own per-time-step residual store through `convergenceControl`, exactly as `SIMPLE` and `PIMPLE` do, so the criterion means the same thing as in the segregated solver. |

Rules that bite:

- **Exact field names only - no regular expressions.** `"(k|epsilon)" 1e-4;`
  is valid in the `SIMPLE` dictionary but **silently never matches** here.
  Write `k` and `epsilon` on separate lines. The solver echoes the segregated
  controls it picked up at start-up (section 6.1); if the echo is missing or
  wrong, the keys are wrong.
- **Fields that are never solved are never checked.** Leaving `k` and
  `epsilon` entries in a laminar case is harmless: the entries are echoed, the
  fields never appear in the residual store, and convergence is decided on the
  coupled blocks alone.
- **The turbulence residual is lagged one outer iteration**, like the
  turbulence itself: the value tested is the one measured at the end of the
  previous outer iteration. Harmless in practice, but the iteration at which
  convergence is declared can be one later than a same-iteration test would
  give. `T` is lagged in exactly the same way and for the same reason.
- **Convergence is never declared on the first outer iteration** (`iter > 1`
  is required), so the initial residuals of 1.0 cannot trigger a false stop on
  a restart from a converged field.
- All tests use **initial** residuals. Final residuals are printed for
  information only (section 6.4).

### 2.4 The shipped `petscOptions` default

Fieldsplit-Schur with hypre BoomerAMG on the Schur block - the first
configuration measured to be both fast and robust on every tested topology:

```
-ksp_type fgmres
-pc_type fieldsplit -pc_fieldsplit_type schur
-pc_fieldsplit_schur_fact_type full
-pc_fieldsplit_schur_precondition selfp
-pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3
-fieldsplit_0_ksp_type preonly
-fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu
-fieldsplit_1_ksp_type preonly
-fieldsplit_1_pc_type hypre
-fieldsplit_1_pc_hypre_type boomeramg
```

The `[u,v,w] / [p]` index sets come free from the block size of 4 that the
solver declares on the matrix, because the unknowns are genuinely block-4
interleaved. Nothing about the split is hard-coded, so any PETSc fieldsplit
configuration is reachable from this string alone.

Using it explicitly - and this is the copy-paste form for **any** options
string in a case dictionary:

```c++
COUPLED
{
    linearSolver    PETSc;

    petscOptions    "-ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition selfp -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type boomeramg";
}
```

The whole string must be one quoted token. OpenFOAM's string reader accepts a
**backslash as the last character on a line** as a continuation, so long
options strings can be broken up for readability - note the space before each
backslash, since the newline is dropped without inserting whitespace:

```c++
    petscOptions
    "-ksp_type fgmres \
     -pc_type fieldsplit -pc_fieldsplit_type schur \
     -pc_fieldsplit_schur_fact_type full \
     -pc_fieldsplit_schur_precondition selfp \
     -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 \
     -fieldsplit_0_ksp_type preonly \
     -fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu \
     -fieldsplit_1_ksp_type preonly \
     -fieldsplit_1_pc_type hypre \
     -fieldsplit_1_pc_hypre_type boomeramg";
```

The default requires a PETSc configured **with hypre**. A PETSc without it
fails at the first solve with an unknown-`PCType` error; either rebuild PETSc
with hypre or select one of the alternatives in section 3.4.

It is a default chosen for robustness on small and two-dimensional cases, and
on a three-dimensional mesh of about a million cells and up it is the wrong
string: section 3.6 gives its replacement and what each part of that is worth.

### 2.5 PETSc options that cannot go through `petscOptions`

Anything consumed *inside* `PetscInitialize` is read before the dictionary
string is inserted and therefore cannot be set from `petscOptions`. In practice
that means `-log_view`, which must come from the environment:

```sh
export PETSC_OPTIONS="-log_view"
mpirun -np 6 -x PETSC_OPTIONS coupledSimpleFoam -parallel
```

`PETSC_OPTIONS` is also a convenient way to try a preconditioner without
editing the case, since PETSc merges the environment string with the
dictionary string.

### 2.6 Keys that do not exist

- `divDevTauCorrection` - removed; ignored if present.
- Any relaxation factor for `p` or `U`. There is none, by construction. Use
  `pseudoTransient` if the run needs damping.
- Non-orthogonal correctors. The non-orthogonal correction is lagged into the
  right-hand side and refreshed every outer iteration; there is no inner
  corrector loop to configure.

---

## 3. Choosing a linear solver and preconditioner

### 3.1 The backends

| `linearSolver` | Library | Parallel | Character |
| --- | --- | --- | --- |
| `SparseLU` | Eigen | **serial only** | Sparse direct LU. Deterministic, needs no tuning, no tolerance to get wrong. Memory and time grow steeply with 3-D fill-in. |
| `BiCGSTAB` | Eigen | **serial only** | Krylov with `IncompleteLUT`. Tuned by `tolerance`, `maxLinearIter`, `dropTolerance`, `fillFactor`. |
| `PETSc` | PETSc | serial and MPI | `MATMPIAIJ` + `KSP`. Everything configurable through `petscOptions`; the only path for 3-D meshes and the only parallel path. |

Requesting an Eigen backend in a parallel run is a fatal error, as is
requesting `PETSc` from a binary built without it. The default already does the
right thing: `SparseLU` in serial, `PETSc` in parallel.

### 3.2 Serial or MPI

Use the serial direct backend when the problem is **2-D and small**, and PETSc
otherwise. This is not a formality - a sparse direct solve is genuinely hard to
beat in 2-D:

| Case | Serial Eigen `SparseLU` | Serial PETSc | np = 4 | np = 6 |
| --- | --- | --- | --- | --- |
| lid-driven cavity 64x64 (16k unknowns) | **0.60 s** | 1.04 s | 0.38 s | - |
| lid-driven cavity 128x128 (66k unknowns) | 3.83 s | - | 2.96 s | 2.15 s |
| lid-driven cavity 256x256 (262k unknowns) | **37.7 s** | 122.9 s | 50.1 s | 41.8 s |

At 262k unknowns in 2-D, `SparseLU` is still the fastest single option. In 3-D
the fill-in explodes and the ranking inverts: a ~12k-cell 3-D duct case takes
74 s with serial `SparseLU` and 2.5 s with PETSc on 4 ranks, at the same 16-17
outer iterations. **PETSc is the 3-D path.**

The PETSc path scales sensibly rather than spectacularly - on the 256x256
cavity, 123 s serial to 50 s on 4 ranks (2.45x, 61 % efficiency) to 42 s on 6
(2.94x, 49 %). The efficiency loss is growth of the fieldsplit iteration count
with the number of subdomains, not communication.

### 3.3 Why the shipped default is what it is

**No preconditioner is universal, and the two obvious candidates are mirror
images of each other.** This table is the reason the default was chosen the way
it was; the columns are the three topology classes in the validation set.

| Configuration | Cavity-class 2-D | 2-D interrupted-fin strip channel | 3-D snappyHexMesh unit cell (1.93M unknowns) |
| --- | --- | --- | --- |
| ASM (overlap 2) / sub-ILU(2) | converges, 78.9 s at 262k unknowns, np6 | **converges everywhere tested**, 2.92 s at np4 | **caps out** - 5,000 inner iterations, ~440 s/outer, unusable |
| fieldsplit-Schur (`selfp`) + GAMG | converges, 42.2 s at 262k unknowns, np6 | **DIVERGES**, serial and parallel - `KSP DIVERGED_ITS` at 5,000 inner iterations, outer loop never converges | converges, 130-470 inner iterations, ~12.2 s/outer |
| **fieldsplit-Schur (`selfp`) + hypre BoomerAMG** | converges | **converges to the serial-direct reference's every printed digit** (25 outer iterations) | **converges, 3.7-3.8 s/outer, ~75 inner iterations** |
| hypre BoomerAMG (scalar) on the whole system | - | diverges at 5,000 inner iterations | - |
| hypre BoomerAMG **nodal** on the whole system | - | converges to the reference (2,361 inner iterations, 11.9 s) | healthy inner solves, but did not close the outer loop under a uniform pseudo-transient recipe; per-inner cost is ASM-class |
| GAMG on the whole block system | - | - | **total failure** - killed at 20 minutes for one iteration |
| LSC | - | converges (21 outer iterations) | **211 s/outer** - unusable |
| Upper-Schur, unsmoothed GAMG + ILU | - | inner solve diverges | - |
| Vanka-style `PCPATCH` | - | requires a `DM`; not reachable from this assembly | - |

Two results carry most of the weight:

- **GAMG on the Schur block diverges on interrupted-fin strip topologies**,
  both serial and parallel, while being perfectly good on cavities and on the
  3-D mesh. ASM/ILU is exactly the opposite. Only the hypre-on-Schur
  configuration passes both.
- **Swapping GAMG for hypre BoomerAMG on the Schur block is worth 2.7x on the
  CPU alone**, from a one-line options change: on the 483k-cell 3-D unit cell
  at np6 the inner iteration count falls from 188 to ~75 and the full run from
  257 s to 91-95 s. Plain scalar AMG applied to the whole stabilised
  saddle-point block system does not work at all.

**The table is about topology, and there is a second axis.** Every row in it
was measured between 262k and 1.93M unknowns, where forming the Schur
approximation is cheap next to the solve it feeds. At 8.6 million unknowns the
shipped `selfp` approximation becomes the single largest cost in the run and
the ranking shifts again; section 3.6 is that regime.

### 3.4 Copy-pasteable option strings

**A. Default - fieldsplit-Schur + hypre BoomerAMG.** Start here for everything.
Requires a PETSc built with hypre.

```
-ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition selfp -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type boomeramg
```

**B. ASM/ILU fallback.** Robust on 2-D topologies and needs no hypre. Use it
when the default's inner solve misbehaves on a 2-D case, or when PETSc has no
hypre. **Do not use it on large 3-D meshes** - it caps out at the inner
iteration limit and each outer iteration costs minutes.

```
-ksp_type fgmres -pc_type asm -pc_asm_overlap 2 -sub_pc_type ilu -sub_pc_factor_levels 2
```

**C. fieldsplit-Schur + GAMG.** Only when hypre is unavailable and the mesh is
3-D or cavity-like. Roughly 2.7x slower than hypre on the Schur block over a
full run, and 3.2x slower per outer iteration on the 483k-cell mesh (section
3.3). **Warning: this configuration diverges on interrupted-fin strip
topologies.**

```
-ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition selfp -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type gamg
```

**D. hypre nodal (systems) AMG on the whole system.** A monolithic block-AMG
alternative to splitting the system, kept as a **diagnostic** rather than as a
fast configuration: it reproduces the reference on the hard 2-D topology, but
each inner solve costs ASM-class iteration counts, and at 8.6 million unknowns
it was measured against the Schur splitting and lost by a factor of three to
four per outer iteration (section 3.7). Worth trying when both A and B fail, to
establish that the assembly is sound, not to make a large case faster.

```
-ksp_type fgmres -pc_type hypre -pc_hypre_type boomeramg -pc_hypre_boomeramg_nodal_coarsen 1 -pc_hypre_boomeramg_vec_interp_variant 1 -pc_hypre_boomeramg_numfunctions 4
```

`-pc_hypre_boomeramg_numfunctions 4` is not optional: without it hypre runs at
`num_functions = 1` and the two nodal options above quietly do nothing
(section 3.7).

**E. Diagnostic overlay.** Append to any of the above to see what the inner
solve is doing before changing anything else:

```
-ksp_monitor_true_residual -ksp_converged_reason
```

**F. Large three-dimensional cases (about a million cells and up).** The
measured replacement for A at that size: the `a11` Schur approximation, hypre
settings that are actually three-dimensional, and a non-restarted Krylov
method. On a 2.15M-cell case it is 5.5x faster than the same case at a fixed
inner tolerance and PETSc's default AMG settings, and A's `selfp` Schur
approximation alone costs 5.6x more per outer iteration there than the `a11`
this uses. Section 3.6 explains each part and the order to reach for them in;
the dictionary that goes with it is in [doc/BENCHMARKS.md](BENCHMARKS.md)
section 9.2.

```
-ksp_type fbcgs -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper -pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type boomeramg -fieldsplit_1_pc_hypre_boomeramg_strong_threshold 0.5 -fieldsplit_1_pc_hypre_boomeramg_coarsen_type HMIS -fieldsplit_1_pc_hypre_boomeramg_interp_type ext+i -fieldsplit_1_pc_hypre_boomeramg_P_max 4 -fieldsplit_1_pc_hypre_boomeramg_truncfactor 0.3 -fieldsplit_1_pc_hypre_boomeramg_agg_nl 1 -fieldsplit_1_pc_hypre_boomeramg_relax_type_all SOR/Jacobi
```

Do not carry it down to a small or two-dimensional case: on the shipped
tutorials it is no faster, and the loose adaptive ceiling it is paired with
moves the sixth significant figure of the pressure drop (section 4.7).

### 3.5 Selection procedure

1. Start with the default (A).
2. If the inner solve reports `KSP DIVERGED_ITS` at the iteration cap on a 2-D
   case, switch to ASM/ILU (B).
3. If inner iteration counts are high but converging on a large 3-D mesh,
   confirm hypre is on the Schur block and not GAMG - that is the 2.7x.
4. If the case is RANS, **tighten `relTol` and cap `maxLinearIter`** rather
   than loosening anything (section 4.4).
5. If nothing converges, try nodal hypre (D) before concluding the assembly is
   at fault - as a diagnostic, not as a fast configuration (section 3.7).
6. If the mesh is 3-D and of the order of a million cells or more, do not stop
   at A once it converges. Its `selfp` Schur approximation alone costs 5.6x
   more per outer iteration at 8.6 million unknowns than the tuned
   configuration's `a11` (F, and section 3.6).

Every one of these is a one-line dictionary change and none requires a rebuild.

### 3.6 Large three-dimensional cases

The shipped default was chosen for robustness across every topology in the
validation set, and that choice costs at scale. On a 2.15M-cell (8.6M-unknown)
interrupted-fin duct at ten ranks, the configuration tuned for that size
converges in **203 s**, against 1112 s for the same case run with `a11` alone
at a fixed `relTol 1e-6` - a factor of 5.5 - and against the segregated
solver's 478 s on the same mesh and the same criteria. This is the size at
which the coupled solver stops being an iteration-count argument and becomes a
wall-clock one (section 8.2). The options string is configuration F of section
3.4 and the dictionary that goes with it is in
[doc/BENCHMARKS.md](BENCHMARKS.md) section 9.2; what follows is what each part
of it buys, in the order worth turning the knobs. The wall-clock figures are
successive steps of one measured chain
([doc/BENCHMARKS.md](BENCHMARKS.md) section 9.1) rather than independent
gains, and the chain was walked in a different order from the one recommended
here, which is by leverage.

1. **The Schur approximation, `selfp` to `a11`.** Change this first: it is the
   largest cost in the shipped string at this size. `selfp` forms an explicit
   sparse matrix product for the Schur complement and rebuilds it every outer
   iteration, which costs **555 s per outer iteration** at 8.6 million unknowns
   against **98.7 s** for `a11`, a factor of 5.6; `a11` instead reuses the
   pressure block the assembly already builds. ASM/ILU sits between them at
   112 s. Pair it with `-pc_fieldsplit_schur_fact_type upper`. On smaller or
   highly non-orthogonal meshes `selfp` remains the more robust choice, which is
   why it is still the shipped default ([doc/BENCHMARKS.md](BENCHMARKS.md)
   section 9.1).
2. **Three-dimensional settings for the AMG on the Schur block**, which are the
   single largest lever in the tuned configuration: 556 s to 354 s on the full
   run. PETSc's BoomerAMG default `strong_threshold` is 0.25, a
   *two-dimensional* value; hypre's own guidance for three-dimensional problems
   is 0.5. Marking too many connections strong inflates operator complexity and
   with it the cost of every V-cycle. HMIS coarsening and truncated `ext+i`
   interpolation go with it.
3. **`adaptiveTolerance yes` with `relTolMax` around `3e-2`** (section 4.7):
   1112 s to 710 s, and it also fixed a run that at a fixed `relTol 1e-6` never
   satisfied the convergence test at all.
4. **A non-restarted Krylov method** - `-ksp_type fbcgs`, flexible BiCGStab,
   instead of restarted FGMRES: 665 s to 556 s. At 8.6 million unknowns,
   orthogonalising every step against a 30-vector Krylov basis is itself a
   significant share of the solve.
5. **`pcReuseInterval` last**, because its best value depends on all the
   others. At the untuned V-cycle cost an interval of 3 was worth only 710 s to
   665 s; once the AMG had been made cheap, an interval of 8 took the same case
   from 354 s to **203 s**. The optimum is 8-12 (206 s at 12, 238 s at 16).

The tuned configuration also runs pseudo-transient continuation at `pseudoCo
50` and caps `maxLinearIter` at 500. Both are section 4 concerns rather than
preconditioner ones, and neither is where the speed comes from.

**None of this belongs on a small or two-dimensional case.** The shipped
defaults remain the better choice there: `relTolMax 3e-2` is too loose for a
case whose answer is judged on its sixth significant figure (section 4.7), and
neither preconditioner reuse nor the three-dimensional AMG settings pay when
the hierarchy is cheap to build in the first place.

### 3.7 Monolithic block AMG: measured and rejected

A single AMG hierarchy over the whole block-4 system - the monolithic coupled
multigrid a commercial coupled solver uses - is the obvious structural
alternative to splitting the system, and it was for a long time the obvious
place to look for the next factor. It was built through hypre and PETSc,
measured against the adopted Schur configuration at 8.6 million unknowns, and
it lost. Cost per outer iteration, with the residual after three outer
iterations as the quality check:

| Preconditioner | s per outer iteration | Residual after 3 outer iterations |
| --- | --- | --- |
| **Schur `a11` + BoomerAMG (adopted)** | **98.7** | **0.037** |
| hypre nodal, block interpolation + nodal Schwarz smoother | 363-370 | 0.058-0.061 |
| hypre nodal systems-AMG (Frobenius) | 389 | 0.977 |
| hypre nodal systems-AMG (row sum) | 684 | 0.968 |
| GAMG, unsmoothed aggregation + block Jacobi | - | diverges with NaN on the first application (`KSP` reason -9) |
| best block AMG + diagonal equation scaling | 292 | 0.114 |

The architecture is directionally sound - block interpolation with a block
smoother improved the three-iteration residual sixteenfold over plain nodal
coarsening on the same matrix - but assembled from generic library components it
costs three to four times more per outer iteration than the Schur splitting. A
competitive version would need purpose-built coarsening, interpolation and
smoothers, which is a solver project rather than a configuration change. The
diagonal equation scaling that helps the block AMG also **hurts** the adopted
configuration, taking it from 98.7 s to 196.7 s per outer iteration, so it does
not carry across either.

**Anyone repeating this must pass `-pc_hypre_boomeramg_numfunctions 4`
explicitly.** PETSc does not forward the matrix block size to hypre:
`MatSetBlockSize(A, 4)` only sets the *default value* of an option whose hypre
call is guarded by whether the option was given, so without the option on the
command line or in `petscOptions`, hypre runs at `num_functions = 1` and every
systems-AMG setting silently degenerates into scalar AMG on a saddle-point
system - which does not work at all (section 3.3). All the runs above passed
it, and all of them carried `-options_left`, which reported no unused options.
A block-AMG experiment without both is not measuring what it believes it is.

## 4. Convergence tuning

### 4.1 The mechanism you are tuning

The outer loop is a **Picard iteration with a nonzero initial guess**: each
outer iteration assembles the coupled system around the current `phi`, the
current `nuEff` and the current lagged pressure gradient, solves it, and
rebuilds `phi` from the solution. Two consequences drive every tuning decision:

- **A loose inner solve produces a negligible update.** The initial guess is
  already the previous solution, so if the Krylov solve only reduces the
  residual by a factor of a thousand, the correction it returns is small and
  the outer loop plateaus. This is why the PETSc `relTol` default is `1e-12`
  and not something conventional. Measured on the laminar cavity: `relTol 1e-3`
  gave "did NOT converge" in 4 s, `relTol 1e-6` the same in 13 s, `relTol
  1e-12` converged.
- **Loose inner solves become useful only when paired with damping.** With
  pseudo-transient continuation on, the operator is better conditioned and a
  looser inner solve still produces a meaningful step. Without it, a loose
  inner solve does not converge slowly, it **plateaus**: on the 483k-cell 3-D
  unit cell, inner `1e-4` with no damping ran 80 outer iterations in 185 s and
  stalled at ~6e-6 / 1e-4 without ever meeting the criteria, whereas the same
  inner tolerance with damping converged (and inner `1e-6` with damping
  converged in 21 iterations and 259 s, against 20 iterations and 496 s for
  the tight-inner baseline - **1.9x**).

Loosening the inner tolerance also degrades mass conservation, which is the
other reason not to do it casually: at `relTol 1e-12` the discrete continuity
rows are satisfied to machine precision (`sum|div(phi)|` ~ 1e-15 to 1e-16),
while at `relTol 1e-6` on a RANS case they are satisfied only to ~1.8e-09
(normalised 2.8e-08).

### 4.2 The `relTol` regime table

There is no single right value. These four regimes are measured, and a
recommendation taken from one contradicts another:

| Regime | `relTol` | Other settings | Evidence |
| --- | --- | --- | --- |
| Small laminar, 2-D | `1e-12` (default) | none needed | `1e-3` and `1e-6` both fail to converge on the cavity |
| Large laminar, 3-D | `1e-6` | `pseudoTransient yes` | 21 outer / 259 s against 20 outer / 496 s at `1e-12`; `1e-4` needs 60 outer and gives the gain back |
| Small RANS | `1e-12` | - | at `1e-6` the outer loop does not finish in 1,000 iterations and continuity degrades to 1.8e-09 |
| Large 3-D RANS | `1e-8` | `maxLinearIter 500`, `pseudoTransient yes` | at `1e-12` that matrix needs ~2,340 inner iterations per outer solve (~168 s/outer, intractable) because `nuEff` heterogeneity degrades both the velocity-block preconditioner and the `selfp` Schur approximation |

### 4.3 Pseudo-transient continuation

```c++
pseudoTransient     yes;
pseudoCo            200;
pseudoUref          0;      // auto
```

Per cell the solver adds `V/dtau` to all three momentum diagonals with

```
h    = cbrt(V)
dtau = pseudoCo * h / max(|U|, 0.05*Uref)
```

and adds the matching `V/dtau * U_old` term to the right-hand side, so the
converged answer is unchanged - it is a relaxation device, not a time-accurate
scheme, and it is the coupled-solver analogue of under-relaxation.

- **Lower `pseudoCo` means more damping.** `200` is the shipped value and is
  what the 483k-cell 3-D measurements use. Values around 50 are more damped and
  were measured to give lower momentum residuals at a fixed iteration cap on a
  RANS case, at the cost of more iterations to get there; `50` is also what the
  large-case configuration of section 3.6 runs at.
- **Turn it on when** the case is large and 3-D, when the inner tolerance is
  loosened, when the start-up transient is violent, or when residuals oscillate
  rather than decreasing.
- **Leave it off for** small laminar 2-D cases, which converge in 12-25
  iterations without it.
- **The length scale is `V^(1/3)`**, which under-estimates the in-plane cell
  size on thin 2-D meshes - one more reason it is off by default.
- **Set `pseudoUref` explicitly** when starting from a zero field with a small
  inlet: the automatic `Uref` is the global maximum of `|U|` over cells and
  boundary faces, which is well defined but can make the `0.05*Uref` floor
  meaninglessly small early on.

### 4.4 Recommended starting points

**Laminar, 2-D, up to a few hundred thousand cells** - serial, no tuning:

```c++
COUPLED
{
    nOuterIterations    100;
    linearSolver        SparseLU;
    residualControl { U 1e-5; p 1e-4; }
}
```

**Laminar, 3-D** - PETSc, damping, moderately loose inner solve:

```c++
COUPLED
{
    nOuterIterations    100;
    linearSolver        PETSc;
    relTol              1e-6;
    maxLinearIter       2000;
    pseudoTransient     yes;
    pseudoCo            200;
    residualControl { U 1e-5; p 1e-4; }
}
```

**RANS, small to moderate** - keep the inner solve tight, raise the iteration
cap, add the turbulence fields to `residualControl`:

```c++
COUPLED
{
    nOuterIterations    1000;
    linearSolver        PETSc;
    relTol              1e-12;
    residualControl
    {
        U               1e-5;
        p               1e-4;
        k               1e-4;
        epsilon         1e-4;
    }
}
```

**RANS, large 3-D** - the one regime where the inner solve must be loosened
*and* capped:

```c++
COUPLED
{
    nOuterIterations    1000;
    linearSolver        PETSc;
    relTol              1e-8;
    maxLinearIter       500;
    pseudoTransient     yes;
    pseudoCo            200;
    residualControl { U 1e-5; p 1e-4; k 1e-4; epsilon 1e-4; }
}
```

Each of the three PETSc recipes above pins `relTol` to one measured constant.
Section 4.7 replaces that constant with a schedule, which is worth reaching
for once a case is large enough that the inner solve dominates - the last two
recipes in particular. Add it to, not instead of, `pseudoTransient`.

Above roughly a million cells in three dimensions the inner solve dominates so
completely that the preconditioner, not the dictionary, is the thing to change
first: section 3.6 gives the tuned configuration, the order to turn the knobs
in and what each one was worth, and the complete dictionary is in
[doc/BENCHMARKS.md](BENCHMARKS.md) section 9.2. The laminar 3-D recipe above
remains the right starting point below that size.

### 4.5 What convergence looks like, and what a stall looks like

**Laminar** cases converge in **12-30 outer iterations** and the count is
essentially mesh-independent: 12 for the 64x64, 128x128 and 256x256 cavities
alike, 22 for a laminar backward-facing step, 19-25 for a 2-D interrupted-fin
strip channel, 20-25 for the 483k-cell 3-D unit cell, and 21, 24, 25, 29 across
a four-point refinement sweep from 269k to 2.15M cells (section 8.2). If a
laminar case is
still running at iteration 60, something is wrong with the linear solve, not
with the outer loop - look at the inner solve first.

**RANS** cases converge in **several hundred** outer iterations, and this is
expected rather than a defect. Convergence has two phases:

1. A fast phase that is the coupled solver doing its job - on a RANS
   backward-facing step, `Ux` falls from 1 to 1.7e-03 in **61** outer
   iterations where SIMPLE needs 451.
2. A long plateau at `Ux` ~5e-05 / `p` ~1.7e-04 lasting hundreds of
   iterations, which collapses by two orders of magnitude in the final few
   iterations once `k` and `epsilon` reach ~1e-05.

**The plateau is the lagged turbulence, not the coupled solve.** The coupled
residual is `b - A x` where `A` contains `D_f = V/aP` built from
`nuEff = nu + nut`: while `nut` is still changing, the operator itself changes
between iterations and the residual cannot vanish however well the block system
is solved. Freezing `nut` (restarting with `turbulence off`) makes the coupled
p-U block converge geometrically at ~0.82 per iteration - 40 iterations take
the residuals from 4e-05/1.2e-04 to 3e-08/7e-09. The same floor appears in the
segregated solver, so it is a property of lagged RANS. **Do not tune the linear
solver to fix a RANS plateau; it is not the linear solver.**

**First-order startup.** The usual staged approach works and is worth using on
a hard mesh: run to a loose criterion with `div(phi,U) bounded Gauss upwind`,
then restart from that solution with `bounded Gauss linearUpwind grad(U)` and
the production criterion. Since the solver reads the latest time directory like
any OpenFOAM application, this is just two runs with an edited `fvSchemes`.

### 4.6 What is not worth trying

Three things that look promising and were measured not to be:

- **Warm-starting from a segregated solution saves almost nothing.** Running
  the segregated solver to its production criterion and then starting the
  coupled solver from that field took 142 segregated plus 21 coupled iterations
  to reach a deep tolerance, against 28 coupled iterations from a cold start -
  a saving of about seven iterations for the entire cost of the segregated run.
  **The outer contraction rate, not the starting distance, sets the iteration
  count.**
- **`reusePreconditioner yes` is counterproductive.** On the 483k-cell case it
  cost 392.7 s against 257.3 s with a fresh setup (+53 %); freezing only the
  hierarchy structure cost +66 %. The Picard operator changes fast enough
  between outer iterations that a stale preconditioner costs more inner
  iterations than its setup saved. **This is an argument against permanent
  freezing, not against reuse as such** - `pcReuseInterval` (section 4.7)
  refreshes periodically instead and is the key to reach for.
- **Chasing assembly cost is chasing the wrong 1 %.** Batched COO assembly
  saved ~0.1 s per iteration (0.7 %) over per-entry insertion, and profiling
  puts the assembly handoff at 1 % of wall time. The per-iteration cost is the
  inner Krylov solve plus preconditioner setup - ~70 % of it in the AMG solve
  on the Schur block and the velocity-block ILU factorisation.

### 4.7 Adaptive inner tolerance and periodic preconditioner reuse

Two optional controls attack the same cost - the inner Krylov solve and its
preconditioner setup, which between them are ~70 % of per-iteration wall time
(section 4.6). Both are **off by default and both are PETSc-only**: with
neither key present, or with `adaptiveTolerance no` and `pcReuseInterval 1`
written out explicitly, the run is bit-for-bit what it was before they
existed.

```c++
COUPLED
{
    linearSolver        PETSc;

    adaptiveTolerance   yes;
    relTolMax           1e-1;
    relTolMin           1e-8;
    adaptiveGamma       0.9;
    adaptiveAlpha       1.5;

    pcReuseInterval     3;
}
```

**The schedule.** Section 4.1 is the reason this key exists: a fixed loose
inner tolerance makes the outer Picard correction negligible and the outer
loop *plateaus* rather than converging slowly, while a fixed tight one makes
each outer iteration unaffordable. The Eisenstat-Walker forcing term resolves
that by making the accuracy demanded of the linear solve track how far the
*nonlinear* iteration still has to go. With `r_n` the outer nonlinear residual
at iteration `n`, the inner relative tolerance is

```
eta_1 = relTolMax
eta_n = adaptiveGamma * (r_n / r_{n-1})^adaptiveAlpha          n >= 2

  safeguard:  if adaptiveGamma*eta_{n-1}^adaptiveAlpha > 0.1
                  eta_n = max(eta_n, adaptiveGamma*eta_{n-1}^adaptiveAlpha)

  clamp:      eta_n = min(relTolMax, max(relTolMin, eta_n))
```

and is handed to `KSPSetTolerances` as the `rtol` for that one solve. The
safeguard is the standard guard against over-loosening: it stops a single
lucky outer iteration from throwing away the accuracy the next one needs,
and it is inactive once the forcing terms are small.

**`r_n` is not a new measure.** It is the same normalised block residual the
`residualControl` test is applied to - the `Initial residual` column of the
log - reduced to one number by taking the worst of the four blocks
(`Ux`, `Uy`, `Uz`, `p`). The schedule is therefore tied to exactly the
quantity the run is judged converged on, and no second residual definition
exists to drift away from the first.

**It is visible in the log.** When the switch is on, each outer iteration
prints one extra line *before* the `Solving for` block:

```
adaptive inner tolerance: relTol = 0.1, nonlinear residual = 1, previous = -1
```

`previous = -1` means "not measured yet", i.e. the first outer iteration.
The line is deliberately not in the `Solving for` format, so `foamLog` and
PyFoam residual parsing are unaffected; with the switch off the line is not
printed at all and existing logs are unchanged byte for byte.

**The Eigen backends ignore it.** `SparseLU` is a direct solve with no
relative tolerance to set, and the Eigen `BiCGSTAB` path takes its tolerance
from `tolerance` rather than `relTol`. Setting `adaptiveTolerance yes` on a
serial run that has not asked for `linearSolver PETSc` is therefore harmless
but does nothing - the schedule is computed and printed, and the backend
discards it. **To exercise the feature in serial, name `linearSolver PETSc`
explicitly**, since the serial default is `SparseLU`.

**Interaction with `relTol`, and what the two tolerances are measured
against.** With the switch on, the dictionary `relTol` is no longer read by
the solve: every solve gets its `eta_n` instead. `tolerance` (the absolute KSP
tolerance) and `maxLinearIter` still apply unchanged, and `maxLinearIter` is
still the right thing to cap on a large RANS matrix.

The two are also **relative to different things**, which is not cosmetic:

| | Convergence test |
| --- | --- |
| fixed `relTol` (default) | PETSc's default, `\|\|r_k\|\| <= relTol*\|\|b\|\|` |
| `adaptiveTolerance yes` | `\|\|r_k\|\| <= eta_n*\|\|b - A x_0\|\|` (`KSPConvergedDefaultSetUIRNorm`) |

The outer loop warm-starts every solve from the previous solution, so
`||b - A x_0||` is orders of magnitude below `||b||` once the run is under
way. Under the `||b||`-relative test a loose `rtol` is therefore *already
satisfied before the first Krylov step*, and `KSPSolve` returns having done no
work at all - `No Iterations 0` on every line, the outer residual frozen, the
run burning its whole iteration budget without moving. That is measured, not
hypothetical, and it is the mechanism behind section 4.1's warning about a
fixed loose `relTol`. An Eisenstat-Walker forcing term is defined against the
residual of the current iterate, so the adaptive path selects the
initial-residual-relative test; the fixed path keeps the original meaning and
the original numbers.

The practical consequence is that **`relTolMax 1e-1` and `relTol 1e-1` are not
the same request**. The first asks each solve to reduce its own starting
residual tenfold, which is real work; the second usually asks for nothing.
Do not carry a value across from one key to the other.

**Preconditioner reuse.** `pcReuseInterval N` builds the preconditioner on
outer iterations `1, N+1, 2N+1, ...` and calls
`KSPSetReusePreconditioner(ksp, PETSC_TRUE)` in between, so the intervening
solves reuse the existing factorisation and AMG hierarchy against the freshly
assembled matrix. This is **periodic refresh, not freezing**: on a rebuild
iteration the flag is explicitly set back to `PETSC_FALSE`. That distinction
is the whole point, because permanent freezing is measured to cost +53 %
(section 4.6) - the Picard operator drifts far enough that a preconditioner
kept for the whole run buys more inner iterations than the skipped setups
saved.

The two reuse keys interact as follows:

| `reusePreconditioner` | `pcReuseInterval` | Behaviour |
| --- | --- | --- |
| `no` (default) | `1` (default) | Rebuild every outer iteration. Historical behaviour, and what every validation number in this repository was measured with. |
| `no` | `N > 1` | Rebuild on iterations `1, N+1, 2N+1, ...`, reuse in between. |
| `yes` | anything | Build once on the first solve, freeze for the whole run. `pcReuseInterval` is ignored and a warning says so. |

**Tuning both together.** They are complementary but not independent: a looser
inner tolerance means fewer Krylov iterations per solve, which raises the
share of each outer iteration spent in preconditioner *setup*, which is
exactly what `pcReuseInterval` removes. Raise `pcReuseInterval` only while the
inner iteration count stays roughly flat; if the count climbs on the reuse
iterations, the interval is longer than the operator's drift allows and the
gain is being paid back. Start at `2` or `3`, and expect the useful interval to
grow with the cost of the setup being skipped: on the 2.15M-cell case of
section 3.6, once the Schur-block AMG had been given three-dimensional settings
and every V-cycle was cheap, the optimum moved out to 8-12 (203 s at 8, 206 s
at 12, 238 s at 16).

**`relTolMax` is the setting that matters, more often than not.** The forcing
term only *falls* when the outer loop converges faster than linearly. A Picard
iteration on a well-behaved case converges at a roughly constant contraction
ratio - about 0.5 on `stripChannel` - and then

```
adaptiveGamma * 0.5^adaptiveAlpha = 0.9 * 0.5^1.5 = 0.32
```

every iteration, which the clamp pins to `relTolMax` from start to finish. The
log line shows this plainly: `relTol = 0.1` on all twenty iterations of the
measurement below. The Eisenstat-Walker ratio earns its keep where the outer
convergence is *not* uniform - violent start-up transients, RANS runs whose
residual falls fast and then plateaus - and elsewhere the schedule degenerates
gracefully into "run at `relTolMax`". **Tune `relTolMax` first; treat
`adaptiveGamma` and `adaptiveAlpha` as second-order.**

**Measured, `stripChannel`** (10,920 cells, 43,680 unknowns, laminar, Re 1000,
the shipped tutorial). Baseline is the shipped default `relTol 1e-12`. Serial
PETSc, one run each:

| Setting | Outer | Total inner | Solver s | Inlet `p` | normalised `sum\|div(phi)\|` |
| --- | --- | --- | --- | --- | --- |
| baseline, `relTol 1e-12` | 25 | 4,677 | 12.33 | 0.79284956 | 2.6e-14 |
| `adaptiveTolerance yes` (`relTolMax 1e-1`) | 20 | 338 | 1.46 | 0.792**85061** | 2.4e-09 |
| `adaptiveTolerance yes`, `relTolMax 1e-2` | 22 | 844 | 2.73 | 0.792849**59** | 1.1e-10 |
| `adaptiveTolerance yes`, `relTolMax 1e-3` | 25 | 1,628 | 4.74 | 0.79284956 | 1.1e-11 |
| `adaptiveTolerance yes`, `relTolMax 1e-4` | 25 | 2,111 | 5.95 | 0.79284956 | 8.7e-13 |

and at np4, best of three back-to-back runs:

| Setting | Outer | Total inner | Solver s | Inlet `p` | normalised `sum\|div(phi)\|` |
| --- | --- | --- | --- | --- | --- |
| baseline, `relTol 1e-12` | 25 | 4,701-4,856 | 3.93 | 0.79284956 | 2.5e-14 |
| `adaptiveTolerance yes` | 20-21 | 353-375 | 0.53 | 0.792850xx | 1.2-2.2e-09 |
| `adaptiveTolerance yes`, `relTolMax 1e-3` | 25 | 1,651-1,751 | 1.58 | 0.79284956 | 1.0e-11 |
| `pcReuseInterval 3` alone | 25 | 5,639-5,820 | 4.49 | 0.79284956 | 2.4e-14 |
| `relTolMax 1e-3` + `pcReuseInterval 3` | 25 | 1,802-1,875 | 1.56 | 0.79284956 | 1.1e-11 |

Read that table carefully, because it contains the trap as well as the win:

- **The win is real and large.** At `relTolMax 1e-3` the inner work falls by
  2.9x and the solve by 2.5x at np4, with the outer count, the pressure drop
  to every printed digit, and a comfortably machine-precision-class continuity
  all preserved. Serial PETSc shows the same 2.6x.
- **The default `relTolMax 1e-1` overshoots on this case.** It is the fastest
  row by far - 7.4x on solver time - but the sixth significant figure of the
  pressure drop moves, the figure is no longer even repeatable run to run
  (0.79285002 to 0.79285061 across three np4 runs), and `sum|div(phi)|`
  degrades from 1e-14 to 1e-09. This is not a defect in the schedule; it is
  section 4.1's loose-inner-solve effect arriving through a new door, and it
  is the reason the switch is off by default. **A run whose answer matters
  should tighten `relTolMax` until the pressure drop stops moving** - here
  `1e-3` - and check the assembly-check line, which detects it immediately.
- **`pcReuseInterval` does not pay on a case this small.** On its own it costs
  ~18 % more inner iterations for ~14 % more wall time: the fieldsplit setup
  it skips is cheap here relative to the extra Krylov work a stale
  preconditioner causes. It turns marginally positive only in combination with
  a loosened tolerance, where setup is a larger share of what is left. Its
  target is a large 3-D system whose AMG setup dominates - not this tutorial.
- **`reusePreconditioner yes` remains a bad idea**, and this case reproduces
  the section 4.6 result at small scale: 10,699 inner iterations against
  ~4,800 for a fresh setup, 2.0x the solve time, same answer.

**What must not change.** Both features are convergence-rate devices, not
discretisation changes. The converged answer, the outer-iteration count on a
well-converging case, and the machine-precision `sum|div(phi)|` of the
assembly check must all survive them. If `sum|div(phi)|` degrades, the
schedule has been allowed to run too loose - tighten `relTolMax` (and, if the
forcing term is genuinely active, lower `relTolMin` so the floor allows the
tighter solves the schedule is asking for), exactly as section 4.1 describes
for a fixed loose `relTol`.

---

## 5. Running in parallel

### 5.1 The sequence

```sh
# 1. clean out anything stale from a previous decomposition
rm -rf processor*

# 2. decompose
decomposePar

# 3. run - linearSolver defaults to PETSc in a parallel run
mpirun -np 6 coupledSimpleFoam -parallel

# 4. reconstruct
reconstructPar -latestTime
```

`system/decomposeParDict` is a normal decomposition dictionary. Both `scotch`
and `simple` decompositions are in the validation set and neither changes the
answer.

Two caveats worth being pedantic about:

- **Remove stale mesh sets before decomposing.** `snappyHexMesh` and the
  topological utilities leave cell and face sets under
  `constant/polyMesh/sets`. If the mesh is later regenerated or modified those
  sets no longer describe it, and `decomposePar` will either fail on them or
  propagate stale topology into the processor directories. Delete
  `constant/polyMesh/sets` (and any leftover `processor*` directories) whenever
  the mesh changes.
- **`mpirun` drains stdin.** A shell script piped into an interpreter loses
  everything after the first `mpirun` unless every `mpirun` gets
  `< /dev/null`. This bites scripted validation runs, not interactive ones.

### 5.2 What to expect

**Outer iteration counts are rank-independent** - identical, not off by one:

| Case | Cells | Unknowns | Serial Eigen | np = 4 | np = 6 |
| --- | --- | --- | --- | --- | --- |
| lid-driven cavity 64x64 | 4,096 | 16,384 | 12 | 12 | - |
| lid-driven cavity 128x128 | 16,384 | 65,536 | 12 | 12 | 12 |
| lid-driven cavity 256x256 | 65,536 | 262,144 | 12 | 12 | 12 |
| 2-D interrupted-fin strip channel | 10,920 | 43,680 | 25 | 25 | 25 |
| backward-facing step, laminar | 12,225 | - | 22 | 22 | - |

This is a direct consequence of the assembly emitting global indices: the same
matrix is built regardless of how it is distributed. It is also the most useful
sanity check available - if a parallel run needs a different number of outer
iterations than the serial one on the same case, suspect the decomposition or
the boundary conditions, not the solver.

**Parallel and serial answers agree to 1e-10 or better:**

| Comparison | Centreline max abs(dUx) | All-cell max abs(dU) |
| --- | --- | --- |
| cavity 64x64, np4 vs serial `SparseLU` | 9.00e-12 | 1.21e-11 |
| cavity 128x128, np4 vs serial `SparseLU` | 3.70e-11 | 5.71e-11 |
| cavity 128x128, np6 vs serial `SparseLU` | 3.40e-11 | 5.01e-11 |
| cavity 256x256, np4 vs serial `SparseLU` | 2.05e-10 | 2.09e-10 |
| cavity 256x256, np6 vs serial `SparseLU` | 2.03e-10 | 2.07e-10 |
| cavity 64x64, serial PETSc vs serial Eigen | 9.00e-12 | 9.55e-12 |

Part of that floor is the ASCII `writePrecision 12`, i.e. ~1e-12 on values of
order one. On the 2-D strip channel the parallel pressure drop reproduces the
serial value to **all printed digits** at both np4 and np6.

**Wall-clock scaling** is real but modest, and the useful rank count is
problem-dependent: the 2-D strip channel goes 3.13 s serial, 2.57 s at np4,
1.79 s at np6 (all 25 iterations); a laminar backward-facing step goes 4.51 s
serial to 2.35 s at np4 (both 22 iterations); the 128x128 cavity at np6 takes
6.29 s with ASM/ILU against 2.17 s with fieldsplit - the preconditioner matters
more than the rank count. More ranks is flat-to-better (3.69, 3.73, 3.23
s/outer at 4, 6, 8 ranks on the 483k-cell case).

**RANS parallel counts are close but not bit-identical** to serial - 771
against 779 outer iterations on one backward-facing-step measurement, with
inlet pressures matching to 0.006 %. At these tolerances the PETSc path's
non-reproducible reductions move the final iteration by a few. Laminar cases
with a tight inner tolerance do reproduce exactly.

---

## 6. Reading the log

The log deliberately keeps stock OpenFOAM formatting, so `foamLog` and PyFoam
parse it unchanged.

### 6.1 Start-up

```
Reading field p
Reading field U
Constructing viscosity and momentum transport models
Selecting momentum transport model type RAS
Selecting RAS turbulence model kEpsilon

Additional (segregated) residual controls: k 0.0001 epsilon 0.0001

Pressure level: pinned at global cell 0 to p = 0

Coupled p-U system: 16384 unknowns, backend EigenSparseLU

Starting coupled iteration loop
```

Four lines to check before letting a long run proceed:

1. **The model line.** `Selecting RAS turbulence model kEpsilon` confirms
   `constant/momentumTransport` was read as intended. A laminar case says
   `Selecting momentum transport model type laminar`.
2. **The segregated residual controls.** If `k` and `epsilon` were meant to be
   convergence criteria and this line is missing, the keys are wrong - most
   often a regex that never matches (section 2.3).
3. **The pressure-level line.** Either
   `Pressure level: pinned at global cell 0 to p = 0` or
   `Pressure level: set by fixedValue patch outlet`. A closed domain that
   reports a `fixedValue` patch, or an open domain that reports pinning, means
   the `p` boundary conditions are not what was intended.
4. **The system size and backend.** `16384 unknowns` should be `4*nCells`, and
   the backend should be the one intended - `EigenSparseLU`, `EigenBiCGSTAB` or
   `PETSc`.

With `adaptiveTolerance yes` the start-up block carries one further line,
echoing the schedule that was read from the dictionary:

```
Adaptive inner tolerance (Eisenstat-Walker): relTol in [1e-08, 0.1], gamma 0.9, alpha 1.5
```

That is the quickest confirmation that all four adaptive keys were picked up;
if the line is absent, the switch is off. A `relTolMin` above `relTolMax` stops
the run here rather than being clamped silently.

### 6.2 A per-iteration block

The first iteration of a lid-driven cavity 64x64 at Re 100, serial:

```
Time = 1s

EigenSparseLU:  Solving for Ux, Initial residual = 1, Final residual = 2.00854400236e-15, No Iterations 1
EigenSparseLU:  Solving for Uy, Initial residual = 1, Final residual = 0.747110880419, No Iterations 1
EigenSparseLU:  Solving for Uz, Initial residual = 0, Final residual = 0, No Iterations 1
EigenSparseLU:  Solving for p, Initial residual = 0, Final residual = 1.77009198474e-14, No Iterations 1
time step continuity errors : sum local = 1.28080978574e-16, global = -8.98498470606e-19, cumulative = -8.98498470606e-19
coupled assembly check: sum|div(phi)| = 1.28080978574e-16 (excl. pRef cell 1.2643671557e-16), max|div(phi)| = 1.64426300419e-18, normalised = 1.75162884398e-16
ExecutionTime = 0.05 s  ClockTime = 0 s
```

| Element | Meaning |
| --- | --- |
| `Time = 1s` | The outer iteration index. One outer iteration per time step; `controlDict` does not cap the loop. |
| `EigenSparseLU:` / `PETSc:` | The active backend, printed where OpenFOAM prints the linear-solver name. |
| `Solving for Ux/Uy/Uz` | The three momentum blocks of the **one** coupled solve, reported separately. They are not three separate solves. |
| `Solving for p` | The continuity block of the same coupled solve. |
| `Initial residual` | The normalised residual of the assembled coupled system **before** the solve, `r = b - A x_old`, summed over the rows of that component and normalised per block with the block generalisation of OpenFOAM's `normFactor`. **This is the number convergence is judged on.** |
| `Final residual` | The same measure recomputed after the solve, i.e. how far the *nonlinear* system moved, not the Krylov residual. |
| `No Iterations` | Inner iterations. `1` for the direct backend; the KSP iteration count for PETSc. A PETSc count sitting at `maxLinearIter` every iteration means the inner solve is not converging (section 3.5). |
| `time step continuity errors` | Stock OpenFOAM format, so existing log parsers keep working. |
| `coupled assembly check` | The diagnostic that matters: `sum\|div(phi)\|` over all cells, the same sum excluding the pinned reference cell, `max\|div(phi)\|`, and the sum normalised by `sum\|phi_f\|`. Because `phi` is rebuilt from exactly the expression that went into the matrix, this should sit at machine precision - ~1e-16 normalised - from the first iteration onwards. **A normalised value that is not near machine precision means the inner solve is too loose or a boundary condition is leaking mass.** |

With `adaptiveTolerance yes` one extra line appears immediately above the
`Solving for` block of each iteration:

```
adaptive inner tolerance: relTol = 0.1, nonlinear residual = 1, previous = -1
```

`relTol` is the forcing term `eta_n` handed to that solve, `nonlinear
residual` is `r_n` (the worst of the four `Initial residual` values printed
just below), and `previous` is `r_{n-1}`, shown as `-1` on the first outer
iteration where there is none. Reading it alongside `No Iterations` is the
quickest way to see whether the schedule is paying: the tolerance should
tighten and the inner iteration count should stay bounded. The line is
absent unless the switch is on, and it is not in the `Solving for` format,
so `foamLog` and PyFoam are unaffected either way.

In a RANS run the turbulence solves print their own stock lines
(`smoothSolver: Solving for epsilon, Initial residual = ...`) between the
coupled block and the continuity report, because they go through OpenFOAM's own
solvers. `bounding k` / `bounding epsilon` warnings mean the same thing they
mean in a segregated run; a handful during start-up is normal, thousands
indicates the model is being asked to do something the physics does not
support.

### 6.3 Convergence

When every criterion is met, the solver prints one last residual block with
`No Iterations 0` - no solve was performed, the residuals were merely measured
- then stops and writes. The same cavity case, twelve iterations later:

```
EigenSparseLU:  Solving for Ux, Initial residual = 5.30956157716e-08, Final residual = 5.30956157716e-08, No Iterations 0
EigenSparseLU:  Solving for Uy, Initial residual = 3.81722024593e-06, Final residual = 3.81722024593e-06, No Iterations 0
EigenSparseLU:  Solving for Uz, Initial residual = 0, Final residual = 0, No Iterations 0
EigenSparseLU:  Solving for p, Initial residual = 8.14870195515e-07, Final residual = 8.14870195515e-07, No Iterations 0

COUPLED solution converged in 12 iterations

ExecutionTime = 0.6 s  ClockTime = 1 s
End
```

The reported count **includes** the final check-only iteration. Convergence is
never declared at iteration 1.

If the cap is reached first:

```
COUPLED did NOT converge within 100 outer iterations
```

The final state is written in both cases, so a capped run can be restarted or
post-processed as it stands.

### 6.4 The one misleading number

**A block whose reference response is numerically zero can print a meaningless
`Final residual`.** In the example above, `Uy` reports
`Final residual = 0.747` at the first iteration of a lid-driven cavity while
`Ux` reports 2e-15 for the same solve. This is a normalisation artefact: the
OpenFOAM-style normalisation factor uses `A*xRef` where `xRef` is the
component-wise global average, and for the `v` block at the first iteration of
this case that quantity is numerically zero, so the denominator collapses. The
absolute linear residual is ~1e-15, as `Ux` shows, and the continuity check on
the same line confirms the solve was exact. Stock OpenFOAM shows the same
artefact for near-zero fields.

**Convergence control uses only the initial residuals**, which are clean. Judge
a run on the `Initial residual` column and on the assembly check; treat
`Final residual` as informational.

---

## 7. Post-processing

Nothing special is required. The solver writes standard OpenFOAM time
directories and every normal tool works unchanged:

```sh
foamPostProcess -func "patchAverage(patch=inlet, field=p)" -latestTime
foamPostProcess -func sample -latestTime
paraFoam
```

**Function objects run afterwards, not during.** A `functions { ... }` block in
`controlDict` is inert while `coupledSimpleFoam` runs (section 1.5), so
quantities such as patch averages, forces or sampled lines are obtained by
running `foamPostProcess` over the written time directories once the run has
finished. In practice this costs nothing for a steady solver - the interesting
value is the converged one - but it does mean **there is no per-iteration
monitoring through function objects**. Use the residual and continuity lines in
the log for that instead (section 6.2), which is what makes them worth reading.

**Fields written** into each output time directory:

| Field | Written | Notes |
| --- | --- | --- |
| `p` | always | kinematic pressure |
| `U` | always | |
| `phi` | always | the conservative Rhie-Chow face flux, rebuilt from the same expression that was assembled, so `div(phi)` is zero to machine precision cell by cell |
| `nut`, `k`, `epsilon`, `omega` | RANS only | written by the momentum-transport model exactly as in a segregated run |

Internal working fields are not written: `rAU` (the Rhie-Chow coefficient
`V/aP`) and `Seff` (the effective area vector of the lagged Rhie-Chow term)
exist only in memory.

Because `phi` is conservative to machine precision, flux-based post-processing
- `surfaceFieldValue` sums, mass balances across patches - is exact rather than
approximately satisfied. That is a genuine convenience of this solver and is
worth checking once on any new case: a patch-flux balance that does not close
means a boundary condition is leaking, not that the post-processing is
inaccurate.

---

## 8. Comparing against the segregated solver

### 8.1 Setting up a fair comparison

Use **one case directory** for both solvers. `foamRun` ignores the `COUPLED`
sub-dictionary and `coupledSimpleFoam` ignores `SIMPLE` and
`relaxationFactors`, so the mesh, the `0/` fields, the schemes and the physical
properties are provably identical rather than merely intended to be.

```sh
# one master case, two copies, so the inputs are provably identical
cp -r case case_seg
cp -r case case_cpl

( cd case_seg && foamRun -solver incompressibleFluid > log.segregated 2>&1 )
( cd case_cpl && coupledSimpleFoam                   > log.coupled    2>&1 )

# compare the final states
tools/compare_fields.py case_cpl <time> case_seg <time> 64
```

Note that Foundation OpenFOAM 14 has no standalone `simpleFoam`; the segregated
reference is `foamRun -solver incompressibleFluid` driven by the `SIMPLE`
dictionary.

Fairness rules that actually matter:

1. **Identical stopping criteria.** `SIMPLE/residualControl` and
   `COUPLED/residualControl` must carry the same numbers. Comparing a coupled
   run at `U 1e-5` with a segregated run at `U 1e-3` measures nothing.
2. **Identical schemes.** Same `divSchemes`, same `laplacianSchemes`. A
   `bounded Gauss upwind` coupled run against a `linearUpwind` segregated run
   is a scheme comparison wearing a solver comparison's clothes.
3. **Compare wall clock *and* outer iterations, and report both.** The coupled
   solver's iteration advantage is large and its per-iteration cost is also
   large; either number alone is misleading.
4. **Check the reference's own convergence level before believing a
   difference.** This is the trap that catches most comparisons - see 8.4.

`tools/compare_fields.py` compares the velocity fields of two case/time
directories cell by cell on a structured 2-D mesh, reporting the largest
cell-wise difference anywhere and the profile along the vertical centreline.
That is the tool used for coupled-against-segregated, parallel-against-serial
and backend-against-backend checks.

### 8.2 What the comparison looks like

Laminar suite, serial, identical meshes and identical criteria
(`U 1e-5, p 1e-4`), `Gauss upwind` convection, segregated relaxation
`U 0.7 / p 0.3`:

| Case | Cells | Segregated iters / wall | Coupled iters / wall | Iteration ratio | Wall ratio |
| --- | --- | --- | --- | --- | --- |
| lid-driven cavity 64x64, Re 100 | 4,096 | 797 / 1.57 s | **12** / 0.60 s | 66x | 2.6x |
| lid-driven cavity 128x128, Re 100 | 16,384 | 2,400 / 17.5 s | **12** / 3.80 s | 200x | 4.6x |
| backward-facing step, laminar | 12,225 | 843 / 5.84 s | **22** / 4.51 s | 38x | 1.3x |
| 2-D interrupted-fin strip channel, Re 400 | 10,920 | 208 / 1.00 s | **19** / 2.29 s | 11x | 0.44x |
| 2-D interrupted-fin strip channel, Re 1000 | 10,920 | 195 / 0.84 s | **25** / 3.13 s | 7.8x | 0.27x |

The honest reading in two dimensions: the iteration advantage is real and large
everywhere, and the wall-clock advantage is not universal. On the strip channel
the segregated solver finishes several times sooner in wall clock despite
needing eight times the iterations, because each coupled iteration is far more
expensive and a 10,920-cell system is small enough that the per-iteration cost
is what decides.

**In three dimensions the answer depends on size, and the two solvers cross
over.** The coupled outer count is essentially flat under refinement while
SIMPLE's grows with it, so the ratio moves steadily in the coupled solver's
favour. The same interrupted-fin duct meshed at four resolutions, both solvers
at their best settings, ten MPI ranks throughout, identical criteria
(`U 1e-5, p 1e-4`), every run formally converged:

| Cells | Segregated | Coupled (tuned, section 3.6) | Speed ratio | dp agreement | Field rms |
| --- | --- | --- | --- | --- | --- |
| 269k | 221 it / 13 s | 21 it / 14 s | 0.93x (tie) | 0.18 % | 0.12 % |
| 533k | 296 it / 44 s | 24 it / 35 s | **1.3x coupled** | 0.12 % | 0.090 % |
| 1.07M | 404 it / 149 s | 25 it / 75 s | **2.0x coupled** | 0.063 % | 0.074 % |
| 2.15M | 555 it / 478 s | 29 it / 203 s | **2.3x coupled** | 0.014 % | 0.075 % |

The segregated count grows 221 -> 296 -> 404 -> 555 while the coupled count
moves only 21 -> 24 -> 25 -> 29, so **the crossover on this hardware sits near
300k cells**: below it the two solvers tie or SIMPLE wins, above a million
cells the coupled solver wins by a factor of two or more, and the gap is still
widening at the largest size measured. At 2.15M cells (8.6M unknowns) the
head-to-head is 555 segregated iterations in 478 s against 29 coupled
iterations in 203 s, both reproducible back to back (203 s / 203 s coupled,
480 s / 478 s segregated). The rms velocity difference over all 2,150,400 cells
is 1.26e-3, 0.075 % of the field maximum - the same difference a slow, tightly
solved coupled run of the same case produces (1.28e-3), so the speed is not
bought out of the answer. The agreement between the two solutions in fact
*improves* with refinement, 0.18 % to 0.014 % in pressure drop, which is what
the O(h^2) Rhie-Chow damping attribution of section 8.3 predicts.

**Those 3-D figures are the tuned configuration, not the shipped defaults.**
Measured before the large-case settings of section 3.6 existed, the coupled
solver was 2.2 times behind segregated on the 483k-cell `snappyHexMesh` unit
cell (91-95 s against 41 s, 24 outer iterations against 142), and on the
2.15M-cell duct the same solver takes 1112 s at a fixed inner tolerance with
PETSc's default AMG settings where the tuned configuration takes 203 s. A large
3-D comparison run set up without section 3.6 measures the settings, not the
solver, and will report a solver that is behind rather than ahead.

Where the coupled solver is unambiguously ahead at every size is deep
convergence: on the 483k-cell mesh the segregated solver **fails** to reach
1e-7 in 8,000 iterations and 39 minutes, while the coupled solver reaches it in
28 iterations. It also requires no relaxation factors and its iteration count
does not grow under mesh refinement.

On small 2-D RANS cases the segregated solver wins on wall clock outright - one
backward-facing-step measurement runs 779 coupled outer iterations against
1,997 SIMPLE iterations (2.6x fewer) but takes longer in wall clock, because
each coupled iteration is far more expensive and the lagged turbulence sets the
convergence floor for both solvers.

### 8.3 The `rhieChowFactor` caveat

**Two correctly converged solutions of the same case will not agree to machine
precision, and the reason is a documented discretisation difference rather than
an error in either solver.**

Segregated SIMPLE takes `rAU = 1/UEqn.A()` from the **relaxed** momentum
matrix, so its converged Rhie-Chow damping is effectively `alpha_U * V/aP`. A
coupled solver has no relaxation factor at all, so `D_f = V/aP` is the natural
choice and is the default. The two therefore differ by a Rhie-Chow term of
order `h^2`, which is invisible almost everywhere and becomes visible exactly
where `rAU` jumps hardest between neighbours - at a singular corner, or in a
RANS field where `nuEff` spans orders of magnitude.

Setting `rhieChowFactor` to the reference run's momentum relaxation factor
reproduces the segregated damping and collapses the difference. Quantified in
three independent settings:

| Setting | Difference at `rhieChowFactor 1.0` | With the factor matched |
| --- | --- | --- |
| lid-driven cavity, lid corner cell | 1.40e-2 all-cell max abs(dUx) | 9.60e-4 at `0.7`; centreline difference falls from 8.82e-4 to 1.43e-6 |
| RANS backward-facing step, pressure drop | 0.678 % | 0.075 % at `0.7` - a 9x reduction at essentially the same outer count (762 against 779) |
| 3-D interrupted-fin unit cell, pressure drop | 1.5 % | 0.11 % at `0.85` |

So:

- **For production, leave `rhieChowFactor` at 1.0.** It is the consistent
  choice for a solver with no relaxation, and every converged coupled
  configuration lands within 0.04 % of every other regardless of preconditioner
  or tolerance.
- **For a verification comparison against a specific segregated run**, set it
  to that run's momentum relaxation factor. The remaining difference is then a
  second, much smaller formulation difference - OpenFOAM's `phiHbyA` uses
  `fvc::flux(rAU*grad(p))` with `rAU` inside the interpolation, whereas the
  coupled assembly uses `D_f * (interpolate(grad p) . Seff)`; these agree to
  order `h^2` and differ most where `rAU` jumps between neighbours.

### 8.4 Do not compare against a loosely converged reference

SIMPLE's residuals decay at roughly 0.993 per iteration near convergence, so
stopping at `U 1e-5` leaves about 1e-3 of *solution* error in the reference.
The measured consequence on the lid-driven cavity:

| Comparison | Centreline max abs(dUx) | All-cell max abs(dUx) |
| --- | --- | --- |
| coupled vs segregated at `U 1e-5 / p 1e-4` (755 SIMPLE iterations) | 8.82e-4 | 1.40e-2 (lid corner cell) |
| coupled vs segregated at `U 1e-9 / p 1e-9` (2,021 SIMPLE iterations) | **1.03e-5** | 1.40e-2 (lid corner cell) |
| coupled with `rhieChowFactor 0.7` vs the tight reference | **1.43e-6** | 9.60e-4 |

The difference against the loose reference is a hundred times larger, and
essentially all of it is the reference's own remaining error. Tightening the
128x128 cavity reference likewise collapses the centreline difference to
2.1e-5.

The same caution applies to integral quantities. Pressure-drop differences at
production `residualControl` are 0.17-0.38 % across the laminar 2-D suite and
0.11 % on the 3-D unit cell once the Rhie-Chow damping definition is matched -
but on a RANS backward-facing step the loosely converged segregated reference
is itself 0.06 % from converged, which is the same size as the remaining
coupled-against-segregated gap. **Below about 0.1 %, judging the coupled solver
against a loosely converged SIMPLE run measures the reference, not the
solver.**

For orientation on the cavity, the segregated result matches the standard
Ghia et al. lid-driven cavity benchmark at Re 100 (u_min ~ -0.199 at y ~ 0.45,
u ~ 0.63 at y ~ 0.945), and the coupled result matches the segregated one to
1.03e-5 along the centreline.

---

## 9. Heat transfer

### 9.1 What is solved

With `COUPLED/solveEnergy yes` the solver adds one segregated step to each
outer iteration, after the coupled p-U solve and after the turbulence update:

```c++
fvScalarMatrix TEqn
(
    fvm::div(phi, T)
  - fvm::laplacian(alphaEff, T)
);

TEqn.relax();     // no-op unless relaxationFactors/equations has a T entry
TEqn.solve();     // uses solvers { T { ... } }, like k and epsilon
```

with

```
alphaEff = nu/Pr + nut/Prt          [m2/s]
```

Three properties of that arrangement matter more than the equation itself.

**It uses the conservative flux, not an interpolated velocity.** `phi` is the
Rhie-Chow face flux the coupled system produced, rebuilt from exactly the
expression the continuity rows contain, so `div(phi)` is zero to machine
precision cell by cell (section 6.2). The energy equation is therefore
discretely conservative: a constant `T` with constant boundary values is
preserved exactly - which is what an adiabatic duct with a fixed inlet
temperature is, and it is checked as such in
[doc/VALIDATION.md](VALIDATION.md) 7.4 - and the boundary heat balance closes
to the level the linear solve was asked for. Building a flux from
`fvc::interpolate(U) & Sf` instead would break both, and the
`coupled energy check` line in the log would show it immediately.

**`nut` is the value the turbulence model has just corrected** on that same
flux, so in RANS the energy step is one step further down the same chain rather
than being lagged separately. In a laminar case `nut()` is identically zero and
`alphaEff` is exactly `nu/Pr`.

**`T` is checked by `residualControl` one outer iteration in arrears**, through
the same mechanism the turbulence fields use (section 2.3). Add a `T` key to
`residualControl` to have it hold up convergence; without one, convergence is
decided on the coupled blocks and whatever other fields are listed.

### 9.2 The passive-scalar assumption, and what it excludes

For **constant properties and no buoyancy**, temperature does not appear in the
momentum equation at all: the momentum and continuity equations close on `U`
and `p` alone, and `T` is advected and diffused by a flow field that does not
depend on it. Solving `T` after the flow, segregated, is then not an
approximation - it is the exact structure of the problem. Nothing computed in
the energy step re-enters the coupled assembly, which is why enabling the
switch cannot change a pressure or a velocity anywhere.

That is also precisely the limit of what is supported here:

- **No buoyancy.** There is no gravity term, no Boussinesq source and no `beta`
  key. A case where the Richardson number is not small - natural convection,
  mixed convection, a stratified duct - is outside this model, and the solver
  gives no warning, because nothing in a passive-scalar run can detect it. The
  answer will look plausible and be wrong.
- **Constant properties.** `nu` comes from the viscosity model and is not a
  function of `T`. Temperature-dependent viscosity, density or conductivity are
  not modelled; for a gas over a large temperature range this is the first
  assumption to fail.
- **Incompressible, so no compression work and no viscous heating.** There is
  no `p dV` term and no dissipation source. Viscous heating matters only at
  large Brinkman number (highly viscous flow, small passages, large velocity),
  but when it does matter it is missing entirely.
- **Kinematic units throughout.** `alphaEff` is a thermal diffusivity in m2/s
  and the reported heat flux is `alphaEff*grad(T)` integrated over the patch,
  i.e. a heat flux divided by `rho*cp`, in K m3/s. Multiply by `rho*cp` to get
  watts. This mirrors the solver's kinematic pressure and is the usual
  incompressible-OpenFOAM convention.
- **The turbulent heat flux is the simple gradient-diffusion hypothesis.**
  `nut/Prt` with a constant `Prt` is the standard engineering closure and is
  what OpenFOAM's own `scalarTransport` function object does, but it carries no
  thermal wall function: at a wall the turbulent diffusivity is the wall
  function's `nut` divided by `Prt`, not a Kader-type thermal law of the wall.
  For a wall-resolved RANS mesh this is fine; for a wall-function mesh in
  forced convection expect a wall heat flux that is systematically off, and
  calibrate against data before trusting it.
- **No conjugate heat transfer, no radiation, no thermal `fvModels`.** A solid
  region, a volumetric heat source and a radiation model are all absent.

If any of those matter, this is the wrong solver: use a thermal solver module
under `foamRun`.

### 9.3 Enabling it

Add three keys to `COUPLED` and a `0/T` field. This is the whole change:

```c++
COUPLED
{
    // ... existing keys unchanged ...

    solveEnergy      yes;
    Pr               0.71;      // laminar Prandtl number
    Prt              0.85;      // turbulent Prandtl number, RANS only

    residualControl { U 1e-5; p 1e-4; T 1e-6; }
}
```

and, alongside the existing `p` and `U` entries,

```c++
solvers
{
    T
    {
        solver          PBiCGStab;
        preconditioner  DILU;
        tolerance       1e-12;
        relTol          0;
    }
}
```

`fvSchemes` needs a `div(phi,T)` entry, and a `laplacianSchemes` `default`
covers `laplacian(alphaEff,T)` (section 1.3). `relaxationFactors { equations
{ T 0.9; } }` is available and behaves exactly as it does for `k` - it is not
normally needed, because the energy equation is linear once `phi` is fixed.

`tutorials/heatedDuct` is a complete worked case.

### 9.4 Thermal boundary conditions

`T` is an ordinary `volScalarField`, so every `fvPatchField` type works; the
solver never branches on a boundary-condition type. The useful ones:

| Condition | Patch entry | Notes |
| --- | --- | --- |
| Isothermal wall | `fixedValue` | The common case. Reproduces `Nu = 7.54` on the laminar parallel-plate benchmark ([doc/VALIDATION.md](VALIDATION.md) section 7) |
| Adiabatic wall | `zeroGradient` | Contributes exactly zero to the wall heat flux and to the energy balance |
| Prescribed heat flux | `fixedGradient` | `gradient` is the **outward** normal gradient `snGrad(T)`, so a positive value heats the fluid. The kinematic flux into the fluid is `alphaEff*gradient` per unit area; for a wanted `q''` in W/m2 set `gradient = q''/(rho*cp*alphaEff)` |
| Robin / external resistance | `mixed` | `valueFraction` blends `refValue` and `refGrad` in the usual way |
| Inlet | `fixedValue`, or `inletOutlet` where backflow is possible | |
| Outlet | `zeroGradient`, or `inletOutlet` | |
| Symmetry, slip, empty | the matching constraint type | Carry zero heat flux automatically, through the patch field's own gradient coefficients - no special case in the solver, and none needed. Verified for `symmetry`: a half-channel with a symmetry plane on the centreline reproduces the full channel cell for cell ([doc/VALIDATION.md](VALIDATION.md) section 7.4) |
| Wedge (axisymmetric) | `wedge` | Takes the same generic path and is zero-flux for a scalar by construction, but **no axisymmetric case has been run**, for the energy equation or for the flow (section 10 of [doc/VALIDATION.md](VALIDATION.md)) |

**`externalWallHeatFluxTemperature` is not available.** It belongs to the
thermophysical transport framework and needs a `kappaEff` from a thermo model,
which a passive-scalar run does not have; constructing it fails at start-up.
Its two useful modes have direct equivalents here: fixed power or flux is
`fixedGradient` as above, and a heat-transfer-coefficient condition
(`q = h*(Ta - Tw)`) is `mixed` with
`valueFraction = h/(h + rho*cp*alphaEff*deltaCoeffs)` and `refValue = Ta`.

### 9.5 What the log shows

With the switch on, start-up prints

```
Energy transport: solving T segregated after the coupled p-U system,
    alphaEff = nu/Pr + nut/Prt with Pr = 7, Prt = 0.85
```

each outer iteration prints the standard OpenFOAM residual line for `T`,

```
DILUPBiCGStab:  Solving for T, Initial residual = 3.33060011185e-07, Final residual = 7.31120308968e-13, No Iterations 55
```

and, immediately after it, a permanent verification line:

```
coupled energy check: wall heat flux = 1.7944553015, enthalpy outflow = 1.73999043168, boundary diffusion = 1.73999043164, imbalance = 2.36837395898e-09 %
```

Read it as follows. **`wall heat flux`** is `alphaEff*grad(T)` integrated over
the `wall`-type patches only, positive into the fluid, in K m3/s.
**`enthalpy outflow`** is `sum(phi_b*T_b)` over all uncoupled boundary faces -
outlet minus inlet for a duct. **`boundary diffusion`** is the same diffusive
integral as the first number but over *every* boundary, so it includes the
conduction through a `fixedValue` inlet, which is why it can differ from the
wall figure. **`imbalance`** is the difference between the last two as a
percentage: summing the assembled equation over every cell telescopes the
internal faces away and leaves exactly those two boundary sums, so a converged
solution must balance them. It is the energy counterpart of the
`coupled assembly check` line and is read the same way - a number that grows
across iterations means the setup is wrong, not the flow.

The imbalance settles at the level the `T` linear solve was asked for
(between 1e-11 % and 3e-09 % on every iteration in the tutorial). Two legitimate exceptions: it is large while
the run is still far from convergence, since nothing forces a balance on an
unconverged field; and on a non-orthogonal mesh the explicit non-orthogonal
correction of the laplacian sits outside the identity, exactly as it does for
the pressure, so the floor is the size of that correction rather than roundoff.

### 9.6 Post-processing

`T` is written into every output time directory like any other field. Because
`phi` is conservative to machine precision, flux-weighted post-processing is
exact: a bulk (mixed-mean) temperature at a cross-section is
`sum(phi*T)/sum(phi)` over that surface and needs no correction.

For a duct, `tools/heated_duct_nusselt.py` reads `T`, `U` and the cell centres
written by `foamPostProcess -func writeCellCentres` and reports the local
Nusselt number at chosen stations. It assumes a structured grid, uniform in the
wall-normal direction, one cell thick - `tutorials/heatedDuct` is what it is
written for.


### 9.7 Do not over-solve the temperature equation

The `residualControl` entry for
`T` decides when the outer loop has converged; the linear-solver `tolerance` and
`relTol` only decide how accurately each individual solve is performed. Setting
`T { tolerance 1e-12; relTol 0; }` - correct when validating against an
analytical solution - makes every solve run to machine precision even after the
initial residual has already fallen below the outer target, which cost 62 to 104
inner iterations per outer iteration on a 2.15M-cell case where 16 to 19 were
enough. Treat `T` like `p` and `U`:

```c
T { solver PBiCGStab; preconditioner DILU; tolerance 1e-09; relTol 0.01; }
```

This changed no reported digit of the temperature field and was worth a factor
of 1.3 on the coupled solver. It matters far more to a segregated run, which
repeats the waste once per SIMPLE iteration: the same change was worth a factor
of 2.2 there. See [doc/BENCHMARKS.md](BENCHMARKS.md) section 9.5.


## 10. Troubleshooting index

| Symptom | Likely cause | Action |
| --- | --- | --- |
| Fatal error reading an `IOdictionary` at start-up | `constant/momentumTransport` missing | Add it with `simulationType laminar;` - required even for laminar cases (1.1) |
| Fatal error about `viscosityModel` | `constant/physicalProperties` missing or lacking the keyword | Add `viscosityModel constant; nu ...;` (1.1) |
| Fatal error naming a missing `wallDist` | `kOmegaSST` without a wall-distance method | Add `wallDist { method meshWave; }` to `fvSchemes` (1.3) |
| `linearSolver ... is a serial (Eigen) backend but this is a parallel run` | `SparseLU`/`BiCGSTAB` requested under `mpirun` | Use `linearSolver PETSc`, or drop the key and take the default (3.1) |
| `linearSolver PETSc requested but ... built without PETSc support` | Eigen-only build | Rebuild with PETSc available; see `doc/INSTALL.md` |
| Unknown `PCType hypre` at the first solve | PETSc built without hypre | Rebuild PETSc with hypre, or use options string B or C (3.4) |
| `Eigen SparseLU factorisation failed (singular matrix?)` | No pressure reference: no `fixedValue`-`p` patch and pinning disabled or misplaced | Check the `Pressure level:` start-up line; give an outlet a `fixedValue` `p` or check `pRefCell` (1.2, 6.1) |
| `PETSc KSP diverged, reason ...` every outer iteration | Preconditioner wrong for this topology | Section 3.3 and 3.5 - most often GAMG-on-Schur on a strip-like topology, or ASM on a large 3-D mesh |
| `No Iterations` pinned at `maxLinearIter` | Inner solve not converging | Change the preconditioner first (3.4); raising `maxLinearIter` alone rarely helps |
| A large 3-D run converges but is slower than the segregated solver | The shipped defaults, which are chosen for robustness on small and 2-D cases | Switch to the large-case configuration - `a11` Schur, 3-D AMG settings, adaptive tolerance, periodic reuse (3.6, options string F in 3.4) |
| Outer residuals plateau and never meet the criteria (laminar) | Inner tolerance too loose | Tighten `relTol`, or turn on `pseudoTransient` if it must stay loose (4.1, 4.2) |
| Outer residuals plateau (RANS) | Lagged turbulence, not the linear solver | Expected; raise `nOuterIterations` and let `k`/`epsilon` converge (4.5) |
| `sum\|div(phi)\|` normalised well above ~1e-15 | Inner solve too loose, or a boundary condition leaking mass | Tighten `relTol`; check for a constraint patch not covered by the zero-flux rule (1.2, 6.2) |
| Answer differs from the segregated solver by ~1 % | Rhie-Chow damping definition, or the reference is not converged | Section 8.3 and 8.4 - in that order |
| Convergence declared far later than expected in a RANS case | Turbulence residual keys written as a regex | Write `k` and `epsilon` as separate keys and check the start-up echo (2.3, 6.1) |
| Parallel run needs a different iteration count from serial (laminar) | Decomposition or boundary conditions, not the solver | Counts are rank-independent by construction (5.2) |
| `decomposePar` fails or the parallel run behaves oddly after remeshing | Stale `constant/polyMesh/sets` or leftover `processor*` | Delete both and decompose again (5.1) |
| A piped run script stops silently after the first `mpirun` | `mpirun` drained stdin | Add `< /dev/null` to every `mpirun` (5.1) |
| `controlDict` function objects produce no output | Function objects are triggered from `Time::run()`, which this solver does not call | Run them afterwards with `foamPostProcess` (1.5, 7) |
| `Final residual` looks large while `Initial residual` is tiny | Normalisation artefact on a near-zero block | Ignore it; convergence uses initial residuals (6.4) |
| `solveEnergy is on ... but the case has no readable temperature field` | `0/T` missing | Add `0/T` with the thermal boundary conditions, or set `solveEnergy no` (9.3) |
| Fatal error asking for a solver for `T` | No `solvers { T { ... } }` entry | Add one; `T` is solved segregated, exactly like `k` (9.3) |
| `Cannot find scheme for div(phi,T)` | `divSchemes { default none; }` with no `div(phi,T)` line | Add the entry (1.3) |
| `T` solves but the answer ignores the walls | Wall patch left at `zeroGradient` (adiabatic) | Set the wall condition explicitly (9.4) |
| `coupled energy check` imbalance stays large | The run is not converged, or the mesh is strongly non-orthogonal | Converge further; on a non-orthogonal mesh the floor is the explicit non-orthogonal correction (9.5) |
| `externalWallHeatFluxTemperature` fails to construct | It needs a thermophysical transport model, which a passive-scalar run has not got | Use `fixedGradient` or `mixed` (9.4) |
| Enabling `solveEnergy` changed the velocity or pressure | It cannot; something else in the case changed | `T` is passive and never enters the momentum rows (9.2) |
