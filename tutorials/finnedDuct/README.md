# finnedDuct — 3-D duct with a fin and symmetry planes

The 3-D tutorial. It is deliberately small — twelve thousand cells — because
its point is not size but topology: constraint patches, a half-obstacle cut by
a symmetry plane, and the sharp cost difference between the serial direct
backend and PETSc once the stencil becomes three-dimensional.

## What it demonstrates

- **Zero-flux constraint patches.** `symmetry`, `symmetryPlane`, `wedge` and
  `slip` patches carry *exactly* zero mass flux. Their per-component transform
  coefficients cannot express that on a tilted face — a diagonal-only
  representation leaks mass through the plane — so the continuity row and the
  flux reconstruction treat them like walls. This case is the tutorial that
  exercises that path.
- **An obstacle cut by a constraint patch.** The fin sits on the z = 0
  symmetry plane, so the meshed fin is a *half* fin and `sideWall_lo` loses the
  40 faces the fin covers: 984 faces against 1 024 on the opposite side.
- **Why the linear-algebra backend matters.** Sparse-LU fill-in explodes on a
  3-D stencil. The same case that the serial direct backend grinds through in
  about a minute takes a few seconds on four MPI ranks with PETSc. This is the
  clearest teaching point in the tutorial set, and the reason
  `Allrun-parallel` is the recommended way to run this case.

## Physics and boundary conditions

A rectangular duct, 2 × 0.5 × 0.375 m, with a single rectangular fin block
removed from the interior. Uniform inlet velocity U = 1 m/s, ν = 1e-3 m²/s.
Laminar, incompressible, 3-D.

| Patch | Type | U | p |
|---|---|---|---|
| `inlet` | patch | `fixedValue (1 0 0)` | `zeroGradient` |
| `outlet` | patch | `zeroGradient` | `fixedValue 0` |
| `bottomPlate` | wall | `noSlip` | `zeroGradient` |
| `topPlate` | wall | `noSlip` | `zeroGradient` |
| `sideWall_lo` | symmetry | `symmetry` | `symmetry` |
| `sideWall_hi` | symmetry | `symmetry` | `symmetry` |
| `fins` | wall | `noSlip` | `zeroGradient` |

The outlet pressure is pinned at zero, so the inlet-average pressure *is* the
inlet-to-outlet pressure difference. Both are kinematic — p/ρ, in m²/s².

### Mesh

`blockMeshDict` builds a plain 64 × 16 × 12 block (12 288 cells).
`topoSetDict` marks one fin block as `strips` and everything else as `fluid`;
`subsetMesh -cellSet fluid -patch fins` keeps `fluid` and moves the 240 fin
cells' newly exposed faces onto the `fins` patch.

Verified mesh after subsetting:

| | |
|---|---|
| cells | **12 048** |
| points | 14 203 |
| faces | 38 212 |
| coupled unknowns | **48 192** |

| Patch | Faces | |
|---|---|---|
| `inlet` | 192 | |
| `outlet` | 192 | |
| `bottomPlate` | 768 | |
| `topPlate` | 768 | |
| `sideWall_lo` | 984 | 40 faces short — the half fin sits on this plane |
| `sideWall_hi` | 1 024 | |
| `fins` | 208 | |

**`fins` must already exist in `blockMeshDict`**, declared as a `wall` with an
empty `faces ( )` list, or `subsetMesh -patch fins` has nowhere to put the
exposed faces.

## How to run it

```sh
./Allrun-parallel    # recommended: decomposePar + MPI + reconstructPar, PETSc
./Allrun             # serial, Eigen SparseLU - correct, but about a minute
./Allclean
```

`Allrun-parallel` takes its rank count from `numberOfSubdomains` in
`system/decomposeParDict` (4 as shipped) and needs a PETSc-enabled build.
`./Allrun` needs nothing beyond the bundled Eigen headers, which is why it is
the script the repository-root `Allrun` invokes.

Both scripts finish by reporting the headline number:

```sh
foamPostProcess -latestTime -func "patchAverage(patch=inlet,fields=(p))"
```

For the segregated comparison, on the same mesh, from this same directory:

```sh
foamRun
```

### Two mandatory steps in the mesh recipe

Both are already in the run scripts.

1. **`rm -rf constant/polyMesh/sets` after `subsetMesh`.** `topoSet` writes the
   cellSets there, `subsetMesh` renumbers the mesh underneath them, and
   `decomposePar` then aborts with a core dump reading the stale sets.
2. **Copy `0.orig` to `0` only after meshing.** The mesh utilities need no `0`
   directory at all, and `subsetMesh` will otherwise subset the fields across
   the newly created patch.

## Expected outcome

Verified with OpenFOAM 14 (build `14-7b05503f98a8`), both serial and on 4
ranks:

| Quantity | Serial | 4 ranks | Tolerance |
|---|---|---|---|
| Backend reported | `EigenSparseLU` | `PETSc` | — |
| Coupled outer iterations | **16** | **16** | exact |
| Inlet-to-outlet Δp | **0.5413873** | **0.5413873** | all 7 digits; identical serial and parallel |
| Coupled system size | 48 192 unknowns | 48 192 unknowns | exact |

Segregated reference on the identical mesh and schemes: **72 iterations**,
Δp = **0.54431523** m²/s². The two solvers agree on the pressure drop to
**0.54 %**, with **4.5× fewer outer iterations** for the coupled solve.

`Allrun` and `Allrun-parallel` print these values and warn if the iteration
count differs from 16.

## Runtime

**This is the case where the backend choice is visible in wall clock.** On an
idle machine the serial Eigen SparseLU solve takes about 70 s; the same case on
four MPI ranks with PETSc takes a few seconds. Meshing is around 3 s either
way. Expect several-fold inflation on a loaded machine, and treat both figures
as an order of magnitude rather than a benchmark — `doc/BENCHMARKS.md` is where
measured timings belong.

The ratio is the durable part of that statement. Run back to back on the same
machine under the same load, the serial solve took 271 s and the 4-rank solve
19.5 s: **an order of magnitude, from four ranks.** That is not parallel
efficiency, it is the direct-versus-iterative difference showing up.

The cause is structural, not a tuning problem: a sparse direct factorisation
suffers fill-in that grows far faster on a 3-D stencil than on a 2-D one. The
serial direct backend stays the right default for small 2-D work because it is
exact and needs no tuning; for 3-D, use PETSc.

## Files worth reading

- `0.orig/U` and `0.orig/p` — the `symmetry` entries on `sideWall_lo` and
  `sideWall_hi`, which are what the zero-flux constraint path keys off.
- `system/topoSetDict` — one `boxToCell` block, deliberately extending below
  z = 0 so the fin is cleanly cut by the symmetry plane.
- `system/fvSolution` — the `COUPLED` dictionary. `nOuterIterations` is 60
  here rather than the 100 of the 2-D cases, and there is deliberately no
  `linearSolver` key, so one directory runs both serially and under MPI.
