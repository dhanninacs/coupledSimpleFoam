# stripChannel — 2-D interrupted-fin channel, Re = 1000

The primary regression case for the solver. It is small, it converges to a
single number that is quoted here to every printed digit, and it produces
**identical results in serial and under MPI** — which is the property that
matters most when the parallel path swaps the linear-algebra backend
underneath.

## What it demonstrates

- **Internal obstacles built with `blockMesh` + `topoSet` + `subsetMesh`**,
  including the zero-face patch trick that gives the exposed faces somewhere to
  land.
- **The `fixedValue`-outlet branch of the pressure treatment.** Unlike
  `cavity`, this case has a pressure patch that fixes a value, so the solver
  takes the level from the patch and ignores `pRefCell`/`pRefValue`. It says so
  in the log: `Pressure level: set by fixedValue patch outlet`.
- **Backend-agnostic assembly.** The same case directory runs serially on the
  Eigen SparseLU direct backend and under MPI on PETSc, with no dictionary
  change, and both reach the same answer in the same number of outer
  iterations.
- **Inlet/outlet boundary conditions** with a reference quantity —
  inlet-to-outlet pressure drop — that is easy to check.

## Physics and boundary conditions

A plane channel of height H = 1 m and length 5 m, with three staggered
interrupted fin strips blocked out of the interior. Uniform inlet velocity
U = 1 m/s, ν = 1e-3 m²/s, so Re = UH/ν = 1000. Laminar, incompressible, 2-D.

| Patch | Type | U | p |
|---|---|---|---|
| `inlet` | patch | `fixedValue (1 0 0)` | `zeroGradient` |
| `outlet` | patch | `zeroGradient` | `fixedValue 0` |
| `walls` | wall | `noSlip` | `zeroGradient` |
| `fins` | wall | `noSlip` | `zeroGradient` |
| `frontAndBack` | empty | `empty` | `empty` |

Because the outlet pressure is pinned at zero, the inlet-average pressure *is*
the inlet-to-outlet pressure difference. Both are kinematic — p/ρ, in m²/s².

### Mesh

`blockMeshDict` builds a plain 240 × 48 × 1 block (11 520 cells).
`topoSetDict` marks the three fin strips as the cellSet `strips` and everything
else as `fluid`; `subsetMesh -cellSet fluid -patch fins` keeps `fluid` and moves
the 600 fin cells' newly exposed faces onto the `fins` patch.

Verified mesh after subsetting:

| | |
|---|---|
| cells | **10 920** |
| points | 22 736 |
| faces | 44 130 |
| coupled unknowns | **43 680** |

| Patch | Faces |
|---|---|
| `inlet` | 48 |
| `outlet` | 48 |
| `walls` | 480 |
| `fins` | 324 |
| `frontAndBack` | 21 840 |

**`fins` must already exist in `blockMeshDict`**, declared as a `wall` with an
empty `faces ( )` list. `subsetMesh -patch fins` has nowhere to put the exposed
faces otherwise.

## How to run it

```sh
./Allrun             # serial, Eigen SparseLU
./Allrun-parallel    # decomposePar + MPI + reconstructPar, PETSc
./Allclean
```

`Allrun-parallel` takes its rank count from `numberOfSubdomains` in
`system/decomposeParDict` (4 as shipped) and needs a PETSc-enabled build;
`./Allrun` needs nothing beyond the bundled Eigen headers.

Both scripts finish by reporting the headline number:

```sh
foamPostProcess -latestTime -func "patchAverage(patch=inlet,fields=(p))"
```

For the segregated comparison, on the same mesh, from this same directory:

```sh
foamRun
```

### Two mandatory steps in the mesh recipe

Both are already in `Allrun`; both are easy to lose when adapting the case.

1. **`rm -rf constant/polyMesh/sets` after `subsetMesh`.** `topoSet` writes the
   cellSets there, `subsetMesh` renumbers the mesh underneath them, and
   `decomposePar` then aborts with a core dump reading the stale sets.
2. **Copy `0.orig` to `0` only after meshing.** `subsetMesh` subsets whatever
   fields are in the current time directory; the mesh utilities need no `0`
   directory at all. It costs nothing here and is mandatory in
   `../stripChannelRAS`, so all five tutorials follow the same recipe.

## Expected outcome

Verified with OpenFOAM 14 (build `14-7b05503f98a8`), both serial and on 4
ranks:

| Quantity | Serial | 4 ranks | Tolerance |
|---|---|---|---|
| Backend reported | `EigenSparseLU` | `PETSc` | — |
| Coupled outer iterations | **25** | **25** | exact |
| Inlet-to-outlet Δp | **0.79284956** | **0.79284956** | all 8 digits; identical serial and parallel |
| Final `sum\|div(phi)\|` | 8.6e-15 | ~5e-13 | order of magnitude |

Segregated reference on the identical mesh and schemes: **195 iterations**,
Δp = **0.79590565** m²/s². The two solvers agree on the pressure drop to
**0.38 %**, with **7.8× fewer outer iterations** for the coupled solve.

`Allrun` prints these values and warns if the iteration count differs from 25.

The `sum|div(phi)|` difference between the two backends is expected and is not
a discretisation difference: the serial direct solve is exact to round-off,
while the PETSc iterative solve stops at the shipped default `relTol 1e-12`. Both are far below any
level that affects the answer, as the identical pressure drop shows.

## Runtime

Seconds. On an idle machine the serial solve takes about 3 s and the 4-rank run
about 3 s of wall clock, with meshing around 2 s on top. Both figures inflate
several-fold on a loaded machine — quote them as an order of magnitude, not a
benchmark. `doc/BENCHMARKS.md` is where measured timings belong.

## Files worth reading

- `system/topoSetDict` — three `boxToCell` strips, then `fluid` = everything
  minus the strips.
- `system/blockMeshDict` — the empty `fins` patch declaration.
- `system/fvSolution` — the `COUPLED` dictionary. Note the deliberate absence
  of a `linearSolver` key, which is what lets one directory run both ways.
