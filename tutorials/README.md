# coupledSimpleFoam tutorials

Five self-contained cases for `coupledSimpleFoam`, the steady-state
incompressible solver that assembles momentum and continuity into a single
implicitly coupled linear system (block size 4, unknowns interleaved
`[u, v, w, p]` per cell).

Each case runs on a fresh machine that has OpenFOAM 14 and this solver
installed, and nothing else. Each carries its own `Allrun`, `Allclean` and
`README.md`; three of the flow-through cases also carry an `Allrun-parallel`.

| Case | Cells | Physics | Demonstrates | Serial runtime |
|---|---|---|---|---|
| [`cavity`](cavity/) | 4 096 | lid-driven cavity, Re = 100, laminar | the smoke test; pressure pinning; explicit backend selection | a few seconds |
| [`stripChannel`](stripChannel/) | 10 920 | 2-D interrupted-fin channel, Re = 1000, laminar | internal obstacles, inlet/outlet, `blockMesh` + `topoSet` + `subsetMesh`, MPI | a few seconds |
| [`finnedDuct`](finnedDuct/) | 12 048 | 3-D duct with a fin, Re = 1000, laminar | 3-D, symmetry constraint patches, why the backend matters | about a minute |
| [`stripChannelRAS`](stripChannelRAS/) | 10 920 | the 2-D channel with kOmegaSST | RANS setup: `k`/`omega`/`nut`, wall functions, `wallDist`, segregated residual controls | about a minute |
| [`heatedDuct`](heatedDuct/) | 8 000 | plane channel with isothermal walls, Re = 20, Pr = 7, laminar | `solveEnergy`, thermal boundary conditions, the energy-balance log line, and the Nu = 7.54 physics gate | a few seconds |

## Running them

Source the OpenFOAM environment and build the solver first:

```sh
. /opt/openfoam14/etc/bashrc     # path depends on the installation
./Allwmake                       # from the repository root
```

Then run one case,

```sh
cd tutorials/stripChannel
./Allrun                 # serial
./Allrun-parallel        # MPI, ranks from system/decomposeParDict
./Allclean               # back to a pristine case
```

or all of them, with a pass/fail summary, from the repository root:

```sh
./Allrun
```

The repository-root `Allrun` invokes each case's serial `Allrun` only, so the
whole suite runs without MPI and without a PETSc-enabled build.

Every case `Allrun` exits non-zero if any step fails, prints its headline
result, and flags an iteration count that differs from the reference value, so
the suite doubles as a regression test.

## What every case has in common

**A `COUPLED` dictionary in `system/fvSolution`.** This is the only dictionary
`coupledSimpleFoam` reads for solution control. It sets the outer-iteration
cap, the linear-algebra backend, the pressure reference, pseudo-transient
continuation and the convergence criteria.

**No under-relaxation of p-U.** A coupled solve has no under-relaxation to
apply. The `relaxationFactors` block that every case still carries exists only
so that the segregated reference run works from the same directory.

**A segregated reference in the same case directory.** Each `controlDict` sets
`application coupledSimpleFoam` and also keeps `solver incompressibleFluid`,
so

```sh
foamRun                  # equivalently: foamRun -solver incompressibleFluid
```

solves the identical mesh, schemes and boundary conditions with OpenFOAM's own
segregated SIMPLE. Both answers are quoted in each case README, which makes the
outer-iteration counts directly comparable.

**Fields shipped in `0.orig/`.** Every `Allrun` meshes first and copies
`0.orig` to `0` afterwards. On the three `subsetMesh` cases this ordering is
mandatory (see below); on `cavity` it is merely consistent. It also means a
case can be re-run any number of times without a fresh checkout.

**A cheap correctness check in the log.** Every outer iteration prints

```
coupled assembly check: sum|div(phi)| = ...
```

which stays at the precision the linear solve was asked for: order 1e-15 with
the serial direct backend, order 1e-13 with the PETSc iterative backend at
the default `relTol 1e-12` on the laminar cases, order 1e-9 on the RANS case. A value that
drifts upward across iterations means the block assembly, not the flow, is
wrong. It is the cheapest smoke test in the log.

## Choosing the backend

The `linearSolver` key in the `COUPLED` dictionary selects the backend for the
block system:

| Value | Backend | Parallel |
|---|---|---|
| `SparseLU` | Eigen sparse direct | serial only |
| `BiCGSTAB` | Eigen iterative, IncompleteLUT preconditioner | serial only |
| `PETSc` | `MATMPIAIJ` + KSP | MPI |

**With the key omitted the solver picks `SparseLU` in serial and `PETSc` under
MPI.** The four other cases leave it out for exactly that reason: one
case directory then runs both ways unchanged. `cavity` sets it explicitly,
because it is the case that documents the key — which is also why `cavity` is a
serial tutorial. Naming a serial Eigen backend in a parallel run is a fatal
error with an explicit message, and so is asking for `PETSc` from a build that
has no PETSc.

## Two traps in the `subsetMesh` cases

`stripChannel`, `finnedDuct` and `stripChannelRAS` cut their internal obstacles
out of a `blockMesh` block with `topoSet` and `subsetMesh`. Two ordering rules
follow, and both are already handled in the `Allrun` scripts.

**Delete `constant/polyMesh/sets` after `subsetMesh`.** `topoSet` writes the
`strips` and `fluid` cellSets there. `subsetMesh` then renumbers the mesh
underneath them, and `decomposePar` aborts with a core dump while reading the
stale sets.

**Copy `0.orig` to `0` only after meshing.** `subsetMesh` subsets whatever
fields it finds in the current time directory, and on a newly created patch it
writes `nan` into a turbulence field. Both `decomposePar` and the solver then
die with `wrong token type - expected Scalar, found ... the word 'nan'`.
`blockMesh`, `topoSet` and `subsetMesh` need no `0` directory at all.

A third point is a modelling one rather than a trap: the patch that receives
the exposed faces — `fins` in all three cases — **must already exist** in
`blockMeshDict`, declared as a `wall` with an empty `faces ( )` list.
`subsetMesh -patch fins` has nowhere to put the faces otherwise.

`topoSet` prints a deprecation notice in OpenFOAM 14 (`topoSet has been
superseded by createZones and is now deprecated`). It still works; the
tutorials accept the warning rather than port to `createZones`, which would
require checking every cell count and reference value below again.

## Reading the reference numbers

The four flow-through cases report `areaAverage(inlet) of p`, a **kinematic**
pressure in m²/s² (that is, p/ρ). Their outlet is pinned at `p = 0` by a
`fixedValue` patch, so the inlet average is directly the inlet-to-outlet
pressure difference.

`cavity` has no `fixedValue` pressure patch anywhere — every one of its patches
is a wall — so the solver pins the level itself at `pRefCell` and prints
`Pressure level: pinned at global cell 0 to p = 0`. The other four print
`Pressure level: set by fixedValue patch outlet`, and their `pRefCell` and
`pRefValue` entries are ignored.

## Verified reference values

All five cases were run end to end with OpenFOAM 14 (build `14-7b05503f98a8`),
serially and — for the three that ship an `Allrun-parallel` — on 4 MPI ranks.
Every value below is measured, not estimated.

| Case | Cells | Unknowns | Coupled iterations (serial / 4 ranks) | Δp, m²/s² (serial / 4 ranks) | Segregated SIMPLE |
|---|---|---|---|---|---|
| `cavity` | 4 096 | 16 384 | 12 / — | pressure pinned, no Δp | 755 iterations |
| `stripChannel` | 10 920 | 43 680 | 25 / 25 | 0.79284956 / 0.79284956 | 195 iterations, 0.79590565 |
| `finnedDuct` | 12 048 | 48 192 | 16 / 16 | 0.5413873 / 0.5413873 | 72 iterations, 0.54431523 |
| `stripChannelRAS` | 10 920 | 43 680 | 146 / 140 | 0.79310179 / 0.7931016 | 195 iterations, 0.79615726 |
| `heatedDuct` | 8 000 | 32 000 | 11 / 11 | 13.0918221595 / 13.0918221595 | 177 iterations, 13.1185022898 |

`heatedDuct` additionally reports a wall heat flux of 1.7944553015 K m³/s and a
Nusselt number of 7.5416 at x = 8.975 m against the exact parallel-plate value
of 7.5407.

The two laminar flow-through cases reproduce their pressure drop **to every
printed digit in serial and under MPI**, on different linear-algebra backends.
`stripChannelRAS` does not, and cannot: `nut` changes the operator between
outer iterations, so the small difference between an exact direct solve and a
PETSc solve at `relTol 1e-8` moves the iteration path. Its serial figures are
the reproducible ones.

Each case README states its own tolerance. `cavity`, `stripChannel` and
`finnedDuct` should match their iteration counts exactly; a change there means
the setup or the solver changed.
