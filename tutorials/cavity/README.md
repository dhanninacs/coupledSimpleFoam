# cavity — lid-driven cavity, Re = 100

The smoke test. The smallest case that exercises the whole solver: block
assembly, the serial direct backend, pressure pinning, and convergence on the
`residualControl` criteria. If this case does not run, nothing else will.

## What it demonstrates

- **The pressure-pinning branch.** Every patch of this cavity is a wall, so no
  `fixedValue` pressure patch exists anywhere and the pressure level of the
  coupled system is otherwise undetermined. The solver detects that, pins the
  pressure row of `pRefCell` to `pRefValue`, and says so in the log:
  `Pressure level: pinned at global cell 0 to p = 0`. The other four tutorials
  take the opposite branch.
- **Explicit backend selection.** This is the one case that sets
  `linearSolver` in the `COUPLED` dictionary, rather than leaving the solver to
  choose. That also makes it a serial-only tutorial: see
  [Running it in parallel](#running-it-in-parallel).
- **The outer-iteration advantage over segregated SIMPLE**, on a case whose
  reference solution is a standard benchmark.

## Physics and boundary conditions

A square cavity of side L = 1 m, 2-D, laminar, incompressible. The lid slides
at U = 1 m/s; ν = 0.01 m²/s, so Re = UL/ν = 100.

| Patch | Type | U | p |
|---|---|---|---|
| `movingWall` | wall | `fixedValue (1 0 0)` | `zeroGradient` |
| `fixedWalls` | wall | `noSlip` | `zeroGradient` |
| `frontAndBack` | empty | `empty` | `empty` |

The mesh is a uniform 64 × 64 × 1 block: **4 096 cells**, 8 450 points,
16 512 faces, giving a coupled system of **16 384 unknowns** (4 per cell).

## How to run it

```sh
./Allrun      # blockMesh, copy 0.orig to 0, coupledSimpleFoam, report
./Allclean    # remove every generated file
```

For the segregated comparison, from this same directory:

```sh
foamRun       # controlDict already carries `solver incompressibleFluid`
```

### Running it in parallel

`system/fvSolution` sets `linearSolver SparseLU`, an Eigen serial backend.
Requesting it under MPI is a fatal error. To run this case in parallel, comment
that key out — the solver then selects PETSc automatically — and use the
supplied `system/decomposeParDict`:

```sh
blockMesh && cp -r 0.orig 0
decomposePar
mpirun -np 4 coupledSimpleFoam -parallel
reconstructPar -latestTime
```

The four flow-through tutorials omit `linearSolver` for exactly this reason
and ship a ready-made `Allrun-parallel`.

## Expected outcome

Verified with OpenFOAM 14 (build `14-7b05503f98a8`) and the Eigen SparseLU
backend:

| Quantity | Value | Tolerance |
|---|---|---|
| Coupled outer iterations | **12** | exact; a different count means the setup changed |
| Coupled system size | 16 384 unknowns | exact |
| Final `sum\|div(phi)\|` | 1.3e-16 | order of magnitude; anything below ~1e-12 is machine precision |
| Segregated SIMPLE reference | **755** iterations | exact for the shipped `residualControl` |

That is a **63× reduction in outer iterations** for the same mesh, schemes and
convergence criteria. Outer iterations are not wall-clock: each coupled
iteration solves a system four times the size of a segregated pressure solve.
See `doc/BENCHMARKS.md` for the timing picture.

The `Allrun` script prints these lines and warns if the iteration count differs
from 12.

### Velocity field

The vertical centreline u-profile is the standard diagnostic for this case: u
falls to a minimum of about −0.198 at roughly mid-height, then rises to about
0.33 at y ≈ 0.88 before meeting the lid. The profile has the shape reported by
Ghia, Ghia and Shin (1982) for Re = 100.

Coupled and segregated solutions of this same case agree on the centreline to
**8.8e-4** in max |Δu| against the shipped segregated reference
(`residualControl` p 1e-4 / U 1e-5, 755 iterations). Most of that gap is the
segregated run's own incomplete convergence, not a difference between the
discretisations: against a tightened segregated reference (p 1e-9 / U 1e-9,
2 021 iterations) the same comparison is **1.0e-5**. Quote whichever number
applies to the reference actually being used.

Sample values, coupled versus the 755-iteration segregated reference, on the
vertical centreline:

| y | u (coupled) | u (segregated) |
|---|---|---|
| 0.132812 | −0.08036757 | −0.07982029 |
| 0.507812 | −0.19823946 | −0.19814713 |
| 0.882812 | 0.32986159 | 0.32919279 |

## Runtime

Seconds. On an idle machine the coupled solve takes about 0.6 s and the
segregated reference about 1.6 s; meshing is well under a second. Both figures
inflate several-fold on a loaded machine, so treat them as an order of
magnitude rather than a benchmark.

## Files worth reading

- `system/fvSolution` — the `COUPLED` dictionary, with every key commented.
  Note that `SIMPLE` and `relaxationFactors` sit alongside it untouched: they
  belong to the segregated reference run, and the coupled solver applies no
  under-relaxation to p-U at all.
- `system/fvSchemes` — the `div(devTauCorrFlux)` entry is **vestigial**. The
  solver builds that flux itself with `fvc::div` of a surface field, which is a
  surface integral rather than a Gauss scheme, so no entry is required; it is
  kept only because it is harmless and `foamRun` ignores it.
- `system/controlDict` — `endTime 5000` is only an iteration cap. Both solvers
  stop on `residualControl` long beforehand.
