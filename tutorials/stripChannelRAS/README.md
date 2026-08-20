# stripChannelRAS — the 2-D interrupted-fin channel with kOmegaSST

The RANS tutorial. Geometry, mesh and boundary conditions are identical to
[`../stripChannel`](../stripChannel/); the only change is `laminar` →
`RAS`/`kOmegaSST`. Reading the two cases side by side is the fastest way to see
exactly what a turbulence model adds to a `coupledSimpleFoam` setup.

## What this case does and does not show

**It shows that the RANS path runs and converges**, that the turbulence fields
are picked up, and that their convergence is monitored through the coupled
solver's own `residualControl`.

**It does not show a turbulent flow field.** At Re = 1000 in this geometry the
SST model produces almost no eddy viscosity, and the RANS answer (Δp = 0.79310
m²/s²) sits within **0.03 %** of the laminar answer (0.79285 m²/s²). Treat the
case as a setup demonstration and a regression test, not as evidence that
turbulence matters at this Reynolds number. A higher-Re variant would need
different inlet conditions and a mesh resolved for it.

## How the coupled solver treats turbulence

Momentum and continuity are still assembled and solved as **one block-4 system**.
`k` and `omega` are **not** in that block: they stay segregated and lagged,
solved by OpenFOAM's own `momentumTransport->correct()` once per coupled outer
iteration, exactly as in SIMPLE.

Two consequences follow, and both are visible in the dictionaries.

**The outer loop is no longer a pure Picard iteration on a fixed operator.**
`nut` changes between iterations, so the operator itself changes, and the outer
loop needs hundreds of iterations rather than the 25 of the laminar case.
`nOuterIterations` is 400 accordingly. A RANS case converging in tens of
coupled iterations would be the surprise, not this.

**Turbulence convergence is checked through extra `residualControl` keys.** `U`
and `p` are the coupled block; any *other* key in the `COUPLED` residualControl
sub-dictionary is taken to be a segregated field, and its initial residual is
read from OpenFOAM's own per-iteration residual store — the same store SIMPLE
and PIMPLE use — so the criterion means precisely what it means there. The
solver echoes what it found:

```
Additional (segregated) residual controls: k 0.0001 omega 0.0001
```

## Three things this case needs that the laminar ones do not

1. **`wallDist { method meshWave; }` in `system/fvSchemes`.** kOmegaSST needs a
   wall-distance field. Without this entry the run fails while the turbulence
   model is being constructed; with it the solver prints
   `Selecting patchDistMethod meshWave`.
2. **`k`, `omega` and `nut` in `0.orig/`**, each with a `fins` boundary entry
   for the patch that `subsetMesh` creates.
3. **`div(phi,k)`, `div(phi,omega)` and `div((nuEff*dev2(T(grad(U)))))` in
   `divSchemes`** — note `nuEff`, not `nu`: the effective viscosity now carries
   `nut` as well.

It also needs **`pseudoTransient yes;`** in the `COUPLED` dictionary. A RANS
case is not stable in the coupled solver without pseudo-transient continuation:
it adds a local time-step term, sized from `pseudoCo`, to the block diagonal,
which damps the change in the operator between outer iterations.

## Physics and boundary conditions

The plane channel of `stripChannel`: H = 1 m, length 5 m, three staggered
interrupted fin strips. U = 1 m/s, ν = 1e-3 m²/s, Re = UH/ν = 1000. RAS
kOmegaSST with low-Re wall treatment.

| Patch | Type | U | p | k | omega | nut |
|---|---|---|---|---|---|---|
| `inlet` | patch | `fixedValue (1 0 0)` | `zeroGradient` | `fixedValue 3.75e-3` | `fixedValue 45` | `calculated` |
| `outlet` | patch | `zeroGradient` | `fixedValue 0` | `zeroGradient` | `zeroGradient` | `calculated` |
| `walls` | wall | `noSlip` | `zeroGradient` | `kLowReWallFunction` | `omegaWallFunction` | `nutLowReWallFunction` |
| `fins` | wall | `noSlip` | `zeroGradient` | `kLowReWallFunction` | `omegaWallFunction` | `nutLowReWallFunction` |
| `frontAndBack` | empty | `empty` | `empty` | `empty` | `empty` | `empty` |

Inlet `k = 3.75e-3` m²/s² corresponds to 5 % turbulence intensity at U = 1 m/s;
inlet `omega = 45` s⁻¹ to a mixing length of about 0.1 H. Both are **nominal
inlet-condition choices**, not values calibrated against measurement.

### Mesh

Identical to `../stripChannel`, produced by the same three commands:
**10 920 cells**, 22 736 points, 44 130 faces, **43 680 coupled unknowns**;
patch faces `inlet` 48, `outlet` 48, `walls` 480, `fins` 324, `frontAndBack`
21 840.

## How to run it

```sh
./Allrun-parallel    # recommended: decomposePar + MPI + reconstructPar, PETSc
./Allrun             # serial, Eigen SparseLU
./Allclean
```

For the segregated comparison, on the same mesh, from this same directory:

```sh
foamRun
```

### The `0.orig` rule is not optional here

This is the case that made the rule necessary. Running `subsetMesh` with a `0/`
directory present corrupts `0/nut`: the utility writes `nan` onto the newly
created `fins` patch, and both `decomposePar` and the solver then die with

```
--> FOAM FATAL IO ERROR:
wrong token type - expected Scalar, found on line 48 the word 'nan'
```

`Allrun` therefore meshes with no `0/` present at all — `blockMesh`, `topoSet`
and `subsetMesh` need none — and copies `0.orig` into place afterwards. It also
deletes `constant/polyMesh/sets` after `subsetMesh`, without which
`decomposePar` core-dumps on the stale cellSets.

## Expected outcome

Verified with OpenFOAM 14 (build `14-7b05503f98a8`):

| Quantity | Serial (Eigen SparseLU) | 4 ranks (PETSc) |
|---|---|---|
| Coupled outer iterations | **146** | **140** |
| Inlet-to-outlet Δp | **0.79310179** | **0.7931016** |

Both Δp values agree to 7 significant figures. Unlike the laminar cases,
neither the iteration count nor the last digits of Δp are identical between
serial and parallel: the operator changes with `nut` between outer iterations,
so the small difference between an exact direct solve and a PETSc solve at
`relTol 1e-8` shifts the path slightly, and repeated 4-rank runs vary in the
eighth digit of Δp. Treat the counts as *approximately* 130–150 and Δp as
**0.793102 ± 1e-6**; only a larger departure indicates a real change.

The serial figures are reproducible to every printed digit and are the ones
`Allrun` checks against.

Log lines worth checking:

```
Selecting RAS turbulence model kOmegaSST
Selecting patchDistMethod meshWave
Additional (segregated) residual controls: k 0.0001 omega 0.0001
Coupled p-U system: 43680 unknowns, backend PETSc
```

Segregated reference on the identical mesh, schemes and turbulence model:
**195 iterations**, Δp = **0.79615726** m²/s². The two solvers agree on the
pressure drop to **0.38 %**.

Note that the coupled advantage in outer iterations is much smaller here than
in the laminar cases — 140 against 195, rather than 25 against 195. That is the
honest picture for a RANS case: the segregated turbulence equations set a floor
on how fast the outer loop can converge, and coupling p-U cannot lift it.

And for the comparison the case is really for — the laminar version of the same
geometry, `../stripChannel`, converges in 25 coupled iterations to Δp =
0.79284956. The RANS answer differs from it by 0.03 %.

## Runtime

About a minute. On an idle machine the 4-rank PETSc run takes roughly 9 s and
the serial Eigen SparseLU run roughly 70 s, with meshing around 2 s on top.
Expect several-fold inflation on a loaded machine; these are orders of
magnitude, not benchmarks.

## Files worth reading

- `constant/momentumTransport` — the whole of the model selection.
- `system/fvSchemes` — the `wallDist` entry and the `nuEff` form of the stress
  divergence scheme.
- `system/fvSolution` — the `"(k|omega)"` solver entry, which *is* read by
  `coupledSimpleFoam` (unlike the `p` and `U` entries above it), and the
  `COUPLED` dictionary with `pseudoTransient yes`.
- Diff this case against `../stripChannel` to see the complete laminar → RANS
  delta in one view.
