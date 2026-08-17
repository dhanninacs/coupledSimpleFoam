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
  and CUDA builds, and a build/run troubleshooting table
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
9. [Troubleshooting index](#9-troubleshooting-index)

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

`p` and `U` are `MUST_READ`. `phi` is read if present and computed from `U`
otherwise, so a cold start does not need it; a restart directory written by
either solver will contain it. The turbulence fields and their wall functions
belong to the momentum-transport model, so they are exactly the fields the
segregated solver needs for the same model - stock tutorial `0/` directories
work unchanged.

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
- Convection scheme choice is a normal steady-state decision.
  `bounded Gauss upwind` is the robust default and is what the validation set
  uses; `bounded Gauss linearUpwind grad(U)` is the usual second-order step up
  once a first-order solution exists (section 4.5).

### 1.4 `system/fvSolution`

Keep the `solvers` sub-dictionary: the coupled p-U block does not use it, but
the **turbulence transport equations do** - they are solved by
`momentumTransport->correct()` through OpenFOAM's normal machinery, so `k`,
`epsilon` and `omega` need `solvers` entries and honour `relaxationFactors`
exactly as in a segregated run. The stock tutorial entries are fine.

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
    reusePreconditioner no;
    petscOptions        "...";    // see 2.4 for the shipped default

    // ---- convergence -----------------------------------------------------
    residualControl
    {
        U               1e-5;     // coupled momentum blocks
        p               1e-4;     // coupled continuity block
        k               1e-4;     // any other key = segregated field
        epsilon         1e-4;     // exact field names only, no regexes
    }
}
```

`tolerance` and `maxLinearIter` are shared key names whose **defaults differ by
backend**; only the active backend reads them.

### 2.2 Every key

| Key | Type | Default | Meaning and when to change it |
| --- | --- | --- | --- |
| `nOuterIterations` | `label` | `100` | Cap on outer (Picard) iterations. Laminar cases converge in 12-25; RANS cases need several hundred, so raise it to ~1000 for RANS. The run writes its final state and reports `COUPLED did NOT converge` when the cap is hit. |
| `linearSolver` | `word` | `PETSc` in a parallel run, `SparseLU` in serial | Backend selection. `SparseLU` and `BiCGSTAB` are Eigen and **serial only** - requesting either in a parallel run is a fatal error. Requesting `PETSc` in a binary built without PETSc is a fatal error naming the fix. See section 3. |
| `pRefCell` | `label` | `0` | **Global** cell index (cell 0 lives on rank 0) whose continuity row becomes the identity row when the pressure level is not fixed by a boundary. Change it only to avoid pinning inside a degenerate cell. |
| `pRefValue` | `scalar` | `0` | Value pinned at `pRefCell`. Used only when no `p` patch reports `fixesValue()`. |
| `rhieChowFactor` | `scalar` | `1.0` | Multiplier on the Rhie-Chow coefficient `D_f = V/aP`. `1.0` is the natural coupled choice, since a coupled solve has no relaxation factor. Set it to a segregated reference run's momentum relaxation factor **only** when reproducing that reference exactly for verification - see section 8.3. |
| `pseudoTransient` | `Switch` | `no` | Enable pseudo-transient continuation: adds `V/dtau` to the three momentum diagonals (and hence to `aP`, and hence to `D_f`). A relaxation device, not a time-accurate scheme. See section 4.3. |
| `pseudoCo` | `scalar` | `200` | Target pseudo-Courant number setting the local step `dtau = pseudoCo*h/max(mag(U), 0.05*Uref)` with `h = cbrt(V)`. Lower is more damping. Has no effect unless `pseudoTransient` is on. |
| `pseudoUref` | `scalar` | `0` | Reference velocity for the `0.05*Uref` floor. `0` means auto: the global maximum of `mag(U)` over internal cells and all boundary faces, recomputed each outer iteration. Set it explicitly when the starting field is zero everywhere except a small inlet, so that the floor is meaningful from the first iteration. |
| `tolerance` | `scalar` | Eigen `1e-10`, PETSc `1e-30` | Absolute linear-solver tolerance. Eigen: the `BiCGSTAB` iterative tolerance (unused by `SparseLU`, which is direct). PETSc: the `KSPSetTolerances` absolute tolerance, deliberately set unreachably tight so that `relTol` is what stops the inner solve. |
| `maxLinearIter` | `label` | Eigen `2000`, PETSc `5000` | Inner iteration cap. Unused by `SparseLU`. For large RANS systems, capping this at ~500 and pairing it with a looser `relTol` is the measured recipe (section 4.4). |
| `dropTolerance` | `scalar` | `1e-5` | Eigen `IncompleteLUT` drop tolerance. `BiCGSTAB` path only. |
| `fillFactor` | `label` | `20` | Eigen `IncompleteLUT` fill factor. `BiCGSTAB` path only. |
| `relTol` | `scalar` | `1e-12` | PETSc relative KSP tolerance - **the tolerance that actually terminates the inner solve**. Tight by default because the outer loop is a Picard iteration started from a nonzero initial guess: a loose inner solve makes the linear update negligible and the outer loop plateaus. See section 4.2 for the regime table. |
| `reusePreconditioner` | `bool` | `no` | Freeze the PETSc preconditioner across outer iterations (`KSPSetReusePreconditioner`). **Measured counterproductive** - see section 4.6. Present for experimentation, not for production. |
| `petscOptions` | `string` | the fieldsplit-Schur + hypre string in 2.4 | Free-form PETSc options string, inserted with `PetscOptionsInsertString` after `PetscInitialize`. Everything about the preconditioner, the Krylov method and the GPU device path is set here; no rebuild is ever needed to change it. Also settable through the `PETSC_OPTIONS` environment variable. |
| `residualControl` | `dictionary` | see below | Convergence thresholds. |

### 2.3 `residualControl`

| Key | Default | Meaning |
| --- | --- | --- |
| `U` | `1e-5` | Threshold on all three coupled momentum blocks (`Ux`, `Uy`, `Uz` each tested separately against this one value). |
| `p` | `1e-4` | Threshold on the coupled continuity block. |
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
  give.
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

### 2.5 PETSc options that cannot go through `petscOptions`

`-use_gpu_aware_mpi` and `-log_view` are consumed *inside* `PetscInitialize`,
which runs before the dictionary string is inserted.

- `-use_gpu_aware_mpi 0` is set by the solver itself before `PetscInitialize`.
  A CPU-only PETSc ignores it.
- `-log_view` must come from the environment:

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

The velocity block is not where the time goes: replacing its ILU sub-solve with
a device ILU is worth 0.01 s/outer, while replacing it with plain Jacobi nearly
doubles the inner iteration count and costs 7.5 s/outer.

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
3-D or cavity-like. Roughly 2.6x slower than hypre on the Schur block.
**Warning: this configuration diverges on interrupted-fin strip topologies.**

```
-ksp_type fgmres -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full -pc_fieldsplit_schur_precondition selfp -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type gamg
```

**D. hypre nodal (systems) AMG on the whole system.** A genuine block-AMG
alternative. It reproduces the reference on the hard 2-D topology, but each
inner solve costs ASM-class iteration counts and it has no device
implementation. Worth trying only when both A and B fail.

```
-ksp_type fgmres -pc_type hypre -pc_hypre_type boomeramg -pc_hypre_boomeramg_nodal_coarsen 1 -pc_hypre_boomeramg_vec_interp_variant 1
```

**E. Diagnostic overlay.** Append to any of the above to see what the inner
solve is doing before changing anything else:

```
-ksp_monitor_true_residual -ksp_converged_reason
```

**F. GPU device path.** Append to configuration A. No rebuild of the solver is
needed; the batched COO assembly is already PETSc's GPU assembly API.

```
-mat_type aijcusparse -vec_type cuda
```

Scope this honestly. On the hardware the GPU measurements were taken on - one
workstation-class card with 1/64-rate FP64, no MPS, and an MPI that is not
CUDA-aware - the device path is **numerically exact** (at `relTol 1e-12` it
reproduces the serial direct reference to all printed digits) but not faster:
it ties the CPU on the 483k-cell case (94.3 s on 4 ranks against 94.7 s for the
CPU on 6), is 7 % slower at fixed preconditioner and rank count, and is ~30x
slower on a 43,680-unknown problem. Its rank scaling is **inverted** - 3.55,
4.28, 5.90 s/outer at 4, 6 and 8 ranks - because without MPS each rank opens
its own CUDA context on the one device and the contexts time-slice; the CPU
control over the same range is flat to improving (3.69, 3.73, 3.23). Roughly
70 % of device wall time is preconditioner setup and the Schur-block AMG solve,
and the matrix is rebuilt and re-shipped every Picard step, so there is nothing
to amortise. **These numbers should not be extrapolated to other hardware**;
a datacentre part with 1/2-rate FP64, MPS or one rank per GPU, and a
substantially larger mesh would all change the balance.

### 3.5 Selection procedure

1. Start with the default (A).
2. If the inner solve reports `KSP DIVERGED_ITS` at the iteration cap on a 2-D
   case, switch to ASM/ILU (B).
3. If inner iteration counts are high but converging on a large 3-D mesh,
   confirm hypre is on the Schur block and not GAMG - that is the 2.7x.
4. If the case is RANS, **tighten `relTol` and cap `maxLinearIter`** rather
   than loosening anything (section 4.4).
5. If nothing converges, try nodal hypre (D) before concluding the assembly is
   at fault.

Every one of these is a one-line dictionary change and none requires a rebuild.

---

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
  what the 3-D measurements use. Values around 50 are more damped and were
  measured to give lower momentum residuals at a fixed iteration cap on a RANS
  case, at the cost of more iterations to get there.
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

### 4.5 What convergence looks like, and what a stall looks like

**Laminar** cases converge in **12-25 outer iterations** and the count is
essentially mesh-independent: 12 for the 64x64, 128x128 and 256x256 cavities
alike, 22 for a laminar backward-facing step, 19-25 for a 2-D interrupted-fin
strip channel, 20-25 for the 483k-cell 3-D unit cell. If a laminar case is
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
  iterations than its setup saved.
- **Chasing assembly cost is chasing the wrong 1 %.** Batched COO assembly
  saved ~0.1 s per iteration (0.7 %) over per-entry insertion, and profiling
  puts the assembly handoff at 1 % of wall time with zero device transfers. The
  per-iteration cost is the inner Krylov solve plus preconditioner setup -
  ~70 % of it in the AMG solve on the Schur block and the velocity-block ILU
  factorisation.

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
more than the rank count. On CPU, more ranks is flat-to-better (3.69, 3.73,
3.23 s/outer at 4, 6, 8 ranks on the 483k-cell case). On the GPU device path
the opposite holds; see section 3.4.

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

The honest reading: the iteration advantage is real and large everywhere, the
wall-clock advantage is not universal, and on the 483k-cell 3-D unit cell at
production criteria the best coupled configuration is **2.2x behind**
segregated (91-95 s against 41 s) while using 24 outer iterations against 142.
Where the coupled solver is unambiguously ahead is deep convergence: on that
same mesh the segregated solver **fails** to reach 1e-7 in 8,000 iterations and
39 minutes, while the coupled solver reaches it in 28 iterations. It also
requires no relaxation factors and its iteration count does not grow under mesh
refinement.

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

## 9. Troubleshooting index

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
