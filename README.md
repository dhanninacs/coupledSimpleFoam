# coupledSimpleFoam

A steady-state incompressible solver for OpenFOAM 14 that solves momentum and
continuity as one implicitly coupled linear system, with no pressure-velocity
under-relaxation.

Where the standard SIMPLE algorithm alternates between a momentum predictor and
a pressure correction and needs relaxation factors to hold that alternation
together, `coupledSimpleFoam` assembles velocity and pressure into a single
block system and solves them together. The practical consequence is that outer
iteration counts collapse — 12 to 25 outer iterations on laminar cases where
SIMPLE needs hundreds — and stop growing under mesh refinement, which above
roughly 300k cells turns into a wall-clock advantage as well. The solver
reads ordinary OpenFOAM case directories, writes the ordinary OpenFOAM log
format, and adds exactly one dictionary to `system/fvSolution`.

**Project status.** Version 0.10 · targets OpenFOAM 14 (Foundation) ·
research-grade: the discretisation is verified (see
[doc/VALIDATION.md](doc/VALIDATION.md)), the dictionary API is stable, and the
scope is deliberately narrow - steady, incompressible, laminar or RANS.

## What it does

The unknowns are interleaved per cell as `[u, v, w, p]` with block size 4, so
the global system has 4N rows for N cells. The momentum rows carry the usual
convection and stress operators from an **unrelaxed** `fvVectorMatrix`, plus an
**implicit pressure-gradient block** built with the mesh's own interpolation
weights and OpenFOAM's `valueInternalCoeffs` / `valueBoundaryCoeffs`, so every
boundary condition that `fvc::grad` handles is exact. The continuity row is the
sum of the **Rhie-Chow face fluxes** over the cell's faces, with the velocity
interpolation and the pressure Laplacian both implicit and only the
non-orthogonal correction lagged. Assembly emits backend-agnostic COO triplets
carrying global indices, so the identical assembly feeds a serial direct solver
and a distributed Krylov solver.

Because pressure and velocity are updated simultaneously, there is **no p-U
under-relaxation to tune** — no `relaxationFactors` entry applies to the coupled
block at all. Four properties follow, and they are the reason to use this
solver rather than SIMPLE:

- **Mesh-independent outer iteration counts.** The lid-driven cavity converges
  in 12 outer iterations at 64x64, at 128x128 and at 256x256; SIMPLE needs 755
  and then thousands as the mesh is refined.
- **Nothing to tune.** No relaxation factors to guess, and no divergence when
  a reference implementation of SIMPLE diverges at relaxation 0.85/0.5.
- **Residuals far below what SIMPLE reaches.** On a 483k-cell three-dimensional
  interrupted-fin unit cell the segregated solver never reaches `1e-7` — 8,000
  iterations and 39 minutes, and it plateaus — while the coupled solver gets
  there in 28 outer iterations.
- **Faster in wall clock once the mesh is large.** With the tuned large-case
  configuration the crossover sits near 300k cells: at 2.15 million cells the
  coupled solver converges in 29 outer iterations and 203 s against the
  segregated solver's 555 iterations and 478 s, a factor of 2.3.

Turbulence is handled through OpenFOAM's own `momentumTransport` framework and
is **segregated and lagged** one outer iteration behind the coupled p-U solve,
which is the standard production pattern and also the main limit on RANS
performance (see below).

Heat transfer follows the same pattern and is off by default. With
`solveEnergy yes` a temperature equation
`fvm::div(phi, T) - fvm::laplacian(alphaEff, T)` is solved segregated after the
coupled solve, on the conservative flux that solve produced. For constant
properties and no buoyancy `T` is a **passive scalar** — it never re-enters the
momentum rows — which is why the switch cannot change a velocity or a pressure
anywhere. The tutorial `heatedDuct` reproduces the exact parallel-plate value
Nu = 7.54.

## Key features

- **OpenFOAM 14** (the OpenFOAM Foundation release, openfoam.org). Standard case
  structure, standard `Solving for ... Initial residual = ...` log lines, so
  `foamLog` and PyFoam parse the output unchanged.
- **Laminar and RANS.** Any incompressible model selectable from
  `constant/momentumTransport`; `kEpsilon` and `kOmegaSST` are the models
  actually exercised.
- **Optional heat transfer**, off by default: three keys (`solveEnergy`, `Pr`,
  `Prt`) and a `0/T` field. Passive scalar only — no buoyancy, constant
  properties — and validated against the exact Nu = 7.54 for isothermal
  parallel plates.
- **Two linear-algebra backends.** Eigen `SparseLU` / `BiCGSTAB` for serial
  runs, PETSc `MATMPIAIJ` + `KSP` for MPI. The backend is a runtime keyword, not
  a build variant, and the parallel result reproduces the serial one to 1e-10 or
  better.
- **Runtime-selectable preconditioners.** The whole PETSc options string is a
  dictionary entry. The `[u,v,w] / [p]` fieldsplit index sets come free from the
  block-4 interleaving, so Schur-complement preconditioning needs no code.
- **Adaptive inner tolerance and preconditioner reuse.** `adaptiveTolerance`
  sets the PETSc relative tolerance each outer iteration from the ratio of
  successive nonlinear block residuals (an Eisenstat-Walker forcing term,
  clamped between `relTolMin` and `relTolMax`); `pcReuseInterval` rebuilds the
  preconditioner every N outer iterations and reuses it in between. Both are
  off by default, and together they carry most of the large-case speed-up.
- **Pseudo-transient continuation.** An optional local `V/dtau` on the momentum
  diagonals, for robustness on hard cases in place of relaxation factors.
- **Runs alongside the segregated solver.** `foamRun` ignores the extra
  `COUPLED` sub-dictionary, so one case directory drives both solvers and the
  comparison is exact.

## Status and honest scope

The discretisation is verified rather than merely plausible: discrete continuity
is satisfied to machine precision every iteration (`sum|div(phi)| ~ 1e-16` on
the cavity), parallel runs reproduce serial runs to 1e-10, and the converged
cavity centreline agrees with a tightly converged SIMPLE reference to 1.03e-5.
Five laminar cases, three-dimensional meshes from 269k up to 2.15 million cells,
and two RANS models have been run end to end. What is **not** tested:
wedge/axisymmetric patches, non-conformal patches, every RAS/LES model other
than `kEpsilon` and
`kOmegaSST`, and the Eigen `BiCGSTAB` backend on the shipped tutorials. Heat
transfer is available (`solveEnergy`) but only as a **passive scalar**: no
buoyancy, constant properties, and the laminar case alone is validated. There
is no multiphase, no `fvModels`/`fvConstraints`, no MRF, and no transient
mode — the solver is steady-state by construction.

The performance trade-off is the important part, and it turns on problem size.
The solver wins on **outer iteration count** and on **depth of convergence**
everywhere, and on **wall clock** above roughly 300k cells on this hardware.
Across a four-point sweep of one three-dimensional interrupted-fin geometry,
both solvers at their best settings, ten MPI ranks throughout and every run
formally converged, the segregated iteration count grows with the mesh
(221 -> 296 -> 404 -> 555) while the coupled count barely
moves (21 -> 24 -> 25 -> 29): the two tie at 269k cells, and the coupled solver
is 1.3x faster at 533k, 2.0x at 1.07M and 2.3x at 2.15M. Agreement improves
along the same sweep, the two pressure drops closing from 0.18 % apart on the
coarsest mesh to 0.014 % apart on the finest, where the rms velocity difference
over all 2.15 million cells is 0.075 % of the field maximum. Below the crossover
segregated SIMPLE is still faster, because each coupled iteration costs far more
than a SIMPLE iteration, and on small two-dimensional RANS cases the gap is
wider still; since both solvers floor out against the lagged turbulence, SIMPLE
also stays the production choice for RANS.

That verdict is recent, and it depends on tuning. Untuned, the coupled solver
was 2.2x behind on the clock on the 483k-cell unit cell - 24 outer iterations
and 91-95 s against the segregated solver's 142 iterations and 41 s, six times
fewer iterations and the only one of the two that converges deeply on that mesh
at all - and the first run of the 2.15M-cell case took 1112 s. An adaptive inner
tolerance, preconditioner reuse, a non-restarted Krylov method and
three-dimensional settings for the AMG on the Schur block account for the factor
of 5.5 between that and 203 s; the resulting configuration is section 9.2 of
[`doc/BENCHMARKS.md`](doc/BENCHMARKS.md). It is not the default, because the
settings that win at two million cells cost time on a small or two-dimensional
mesh. Choose this solver for three-dimensional cases of roughly a million cells
and up, when relaxation tuning is failing, when iteration counts grow with mesh
refinement, or when deep residuals matter; choose SIMPLE below the crossover,
where a well-behaved case just needs to finish quickly, and for RANS at
production tolerances.

Full numbers in [`doc/BENCHMARKS.md`](doc/BENCHMARKS.md); the verification
evidence and its acceptance criteria in [`doc/VALIDATION.md`](doc/VALIDATION.md).

## Requirements

| Component | Version | Required | Notes |
| --- | --- | --- | --- |
| OpenFOAM | 14 (Foundation, openfoam.org) | yes | Targets the OpenFOAM 14 API; earlier and ESI releases will not compile unmodified |
| C++ toolchain | as required by the local OpenFOAM | yes | No extra standard or flag is imposed; `wmake` uses OpenFOAM's own compiler settings |
| Eigen | 3.4+ (headers only) | yes | Serial backend. Debian/Ubuntu: `libeigen3-dev` |
| PETSc | 3.19+ | optional | Enables the MPI backend. Debian/Ubuntu: `petsc-dev`. The default preconditioner needs a PETSc configured with hypre |
| Python | 3.8+ | optional | `tools/compare_fields.py` needs the standard library only; `tools/reference_prototype/` additionally needs NumPy and SciPy |

Without PETSc the solver builds and runs serially against Eigen alone. MPI runs
require PETSc. The sources are device agnostic, and an optional CUDA build layer
for NVIDIA GPUs lives in a companion repository,
[`coupledSimpleFoam-gpu`](https://github.com/dhanninacs/coupledSimpleFoam-gpu),
where one GPU is worth roughly ten CPU cores on the reference machine and does
not beat them.

## Quick start

```sh
# 1. Dependencies (Debian/Ubuntu; PETSc is optional but wanted for MPI)
sudo apt-get install -y libeigen3-dev petsc-dev

# 2. Source the local OpenFOAM 14 environment
. /opt/openfoam14/etc/bashrc

# 3. Build from the root of a clone of this repository.
#    The executable lands in $FOAM_USER_APPBIN.
cd coupledSimpleFoam
./Allwmake

# 4. Run the lid-driven cavity tutorial
cd tutorials/cavity
./Allrun
```

`./Allwmake` prints the configuration it chose (OpenFOAM version, Eigen include,
whether PETSc was found) before compiling. To build without PETSc even when it
is installed, use `COUPLED_NO_PETSC=1 ./Allwmake`.

The cavity log should contain, in order:

```text
Pressure level: pinned at global cell 0 to p = 0
Coupled p-U system: 16384 unknowns, backend EigenSparseLU

Time = 1s

EigenSparseLU:  Solving for Ux, Initial residual = 1, Final residual = 2.008544e-15, No Iterations 1
EigenSparseLU:  Solving for Uy, Initial residual = 1, Final residual = 0.7471108, No Iterations 1
EigenSparseLU:  Solving for Uz, Initial residual = 0, Final residual = 0, No Iterations 1
EigenSparseLU:  Solving for p,  Initial residual = 0, Final residual = 1.770092e-14, No Iterations 1
time step continuity errors : sum local = 1.28081e-16, global = -8.98498e-19, cumulative = -8.98498e-19
coupled assembly check: sum|div(phi)| = 1.28081e-16 (excl. pRef cell 1.264367e-16), max|div(phi)| = 1.644263e-18, normalised = 1.751629e-16
...
COUPLED solution converged in 12 iterations
```

Three things to read in that output. **`COUPLED solution converged in 12
iterations`** is the headline: `foamRun -solver incompressibleFluid` from the
same directory converges in 755. The **`coupled assembly check`** line is the
cheapest smoke test that the block assembly is correct — `sum|div(phi)|` should
sit at machine precision (~1e-16 with a direct backend, ~1e-9 with an iterative
one at a loose `relTol`) and stay there. The `Final residual = 0.747` on `Uy` at
the first iteration is a **normalisation artefact**, not a failure: for a field
that is still numerically zero the OpenFOAM-style normalisation factor
collapses. Stock OpenFOAM shows the same artefact, and convergence control uses
only the initial residuals.

`./Allrun` at the repository root runs every tutorial and prints a pass/fail
summary, so it doubles as a regression gate.

## Usage in your own case

The solver reads one sub-dictionary, `COUPLED`, from `system/fvSolution`. Add it
alongside the existing `solvers` and `SIMPLE` entries — `foamRun` ignores it, so
the same case still runs segregated for comparison. Every key has a default; a
case with no `COUPLED` dictionary at all still runs.

```c
COUPLED
{
    nOuterIterations    100;      // outer (Picard) iteration cap

    // SparseLU | BiCGSTAB (Eigen, serial) | PETSc (MPI and serial).
    // Omit the key and the solver picks SparseLU in serial, PETSc under MPI.
    linearSolver        SparseLU;

    pRefCell            0;        // GLOBAL cell index; used only when no
    pRefValue           0;        // p patch reports fixesValue()

    pseudoTransient     no;       // continuation; recommended for RANS and
    pseudoCo            200;      // for large 3-D meshes

    relTol              1e-12;    // PETSc relative KSP tolerance - the key
                                  // most likely to need changing; see below

    residualControl
    {
        U               1e-5;     // the coupled momentum blocks
        p               1e-4;     // the coupled continuity block
        k               1e-4;     // any other key = a segregated field,
        omega           1e-4;     // matched by EXACT name, not by regex
    }
}
```

Three case-level requirements that are easy to miss: every case needs
`constant/momentumTransport` with a `simulationType` entry (`laminar` is
enough) and `constant/physicalProperties` with a `viscosityModel` entry, and
`system/fvSchemes` must set `ddtSchemes { default steadyState; }`. `kOmegaSST`
additionally needs `wallDist { method meshWave; }`.

`relTol` is the one setting that genuinely cannot be set once and forgotten: it
wants `1e-12` on small laminar and small RANS cases, `1e-6` on large laminar
cases with pseudo-transient continuation, and `1e-8` with `maxLinearIter 500` on
large RANS cases. Too loose and the outer Picard loop stalls; too tight on a
large RANS matrix and each outer iteration becomes intractable. On large
three-dimensional cases `adaptiveTolerance yes` takes the choice over, deriving
the inner tolerance from the nonlinear residual history instead, and
`pcReuseInterval` decides how often the preconditioner is rebuilt; both default
to off, and the values that go with them on a million cells and up are in
[`doc/BENCHMARKS.md`](doc/BENCHMARKS.md) section 9.2.

Every key, every default, the shipped PETSc options string, the preconditioner
selection procedure and the full limitation list are in
[`doc/USAGE.md`](doc/USAGE.md).

## Documentation

| Document | Contents |
| --- | --- |
| [`README.md`](README.md) | This file: what the solver is, how to build it, how to try it |
| [`doc/THEORY.md`](doc/THEORY.md) | The coupled formulation: unknown ordering, the implicit pressure-gradient block, the Rhie-Chow continuity row, boundary-condition handling, residual normalisation |
| [`doc/INSTALL.md`](doc/INSTALL.md) | Prerequisites, build switches, and the optional PETSc build |
| [`doc/USAGE.md`](doc/USAGE.md) | The `COUPLED` dictionary key by key, preconditioner and tolerance guidance, and the full list of limitations |
| [`doc/BENCHMARKS.md`](doc/BENCHMARKS.md) | Iteration counts and wall-clock measurements against segregated SIMPLE, MPI scaling, the preconditioner study, the four-point size sweep and the tuned large-case configuration |
| [`doc/VALIDATION.md`](doc/VALIDATION.md) | Verification evidence: discrete continuity, agreement with segregated references, parallel-vs-serial reproduction, regression gates |
| [`doc/DEVELOPMENT.md`](doc/DEVELOPMENT.md) | Code layout, the OpenFOAM 14 API notes the implementation depends on, and how to add a backend |
| [`doc/REFERENCES.md`](doc/REFERENCES.md) | Literature and software references |
| [`CONTRIBUTING.md`](CONTRIBUTING.md) | Build, test and pull-request conventions |
| [`tutorials/`](tutorials/) | Five self-contained cases: `cavity` (laminar, pressure pinning), `stripChannel` (2-D interrupted-fin channel, the primary regression case), `finnedDuct` (3-D duct with a fin and symmetry planes), `stripChannelRAS` (the RANS demonstration), `heatedDuct` (heat transfer, and the Nu = 7.54 physics gate) |
| [`tools/`](tools/) | `compare_fields.py`, a cell-by-cell field comparison for structured 2-D meshes, `heated_duct_nusselt.py`, the Nusselt-number post-processor for `heatedDuct`, and a small independent Python finite-volume implementation of the same algorithm used as a cross-check |

## If it stalls or diverges

Two troubleshooting tables cover the failures seen so far, indexed by symptom:
[doc/INSTALL.md](doc/INSTALL.md) section 8 for build, start-up and
case-setup failures, and [doc/USAGE.md](doc/USAGE.md) section 10 for
convergence behaviour - stalling outer loops, diverging inner solves and
preconditioner choice.

## Repository layout

```text
coupledSimpleFoam/
|-- Allwmake                     build the solver
|-- Allrun                       run every tutorial, print a pass/fail summary
|-- Allclean                     remove build objects and tutorial run output
|-- src/coupledSimpleFoam/
|   |-- coupledSimpleFoam.C      assembly, outer loop, energy, residuals, flux
|   |-- createFields.H           fields, viscosity and momentumTransport models
|   |-- coupledLinearSystem.C/H  backend-agnostic COO system and the factory
|   |-- eigenCoupledSolver.C/H   serial backend: SparseLU, BiCGSTAB
|   |-- petscCoupledSolver.C/H   MPI backend: MATMPIAIJ + KSP
|   `-- Make/{files,options}
|-- tutorials/
|   |-- cavity/                  lid-driven cavity, laminar
|   |-- stripChannel/            2-D interrupted-fin channel, laminar
|   |-- finnedDuct/              3-D duct with a fin, symmetry planes
|   |-- stripChannelRAS/         2-D interrupted-fin channel, kOmegaSST
|   `-- heatedDuct/              3-D duct with isothermal walls, energy on
|-- tools/
|   |-- compare_fields.py        field-by-field comparison of two solutions
|   |-- heated_duct_nusselt.py   Nusselt number from a heated-case result
|   `-- reference_prototype/     independent 2-D Python implementation
|-- doc/                         theory, install, usage, benchmarks, validation,
|                                development notes, references
|-- CHANGELOG.md
|-- CONTRIBUTING.md
`-- LICENSE                      GPL-3.0
```

Each tutorial directory holds `0.orig/`, `constant/`, `system/`, an `Allrun`,
an `Allclean` and a `README.md` recording the expected iteration count and
pressure drop; three of them also carry an `Allrun-parallel`.

The companion repository named under Requirements carries no sources of its own;
it takes this one in as a submodule and adds only a build configuration and its
own documentation.

## Licence

Copyright (C) 2026 dhanninacs.

Distributed under the **GNU General Public License version 3**; see
[`LICENSE`](LICENSE). The solver links OpenFOAM, which is GPL-3.0, so a
compatible licence is required.

## Acknowledgements and trademark

This offering is **not approved or endorsed by OpenCFD Ltd**, producer and
distributor of the OpenFOAM software via www.openfoam.com, **nor by the
OpenFOAM Foundation**, distributor of OpenFOAM software via www.openfoam.org.
OPENFOAM is a registered trademark of OpenCFD Ltd.

The solver builds on OpenFOAM's finite-volume framework, its
`momentumTransport` models and its boundary-condition coefficient interface, and
on Eigen and PETSc for the linear algebra.

## Citing and further reading

The coupled pressure-velocity formulation implemented here follows:

- Darwish, M., Sraj, I., Moukalled, F., *A coupled finite volume solver for the
  solution of incompressible flows on unstructured grids*, Journal of
  Computational Physics **228** (2009) 180-201.
- Moukalled, F., Mangani, L., Darwish, M., *The Finite Volume Method in
  Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and
  Matlab*, Springer, 2016 — Chapter 15 for collocated SIMPLE, the Rhie-Chow
  interpolation and the coupled assembly.
- Uroic, T., Jasak, H., *Block-selective algebraic multigrid for implicitly
  coupled pressure-velocity systems*, Computers & Fluids **167** (2018) 100-110.
- Ghia, U., Ghia, K.N., Shin, C.T., *High-Re solutions for incompressible flow
  using the Navier-Stokes equations and a multigrid method*, Journal of
  Computational Physics **48** (1982) 387-411 — the lid-driven cavity reference
  data used for orientation.

Software references, the PETSc and hypre citations, and further reading are in
[`doc/REFERENCES.md`](doc/REFERENCES.md).
