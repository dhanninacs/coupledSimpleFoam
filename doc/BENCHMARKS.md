# Benchmarks

This file records what the solver has been measured to do, against the stock
OpenFOAM segregated steady incompressible solver, on the cases it has been run
on. It is written to be read by someone deciding whether the solver is worth
using, so the results that go against it are reported in the same detail as
the results that favour it.

Correctness evidence - what the solver has been verified to compute, and what
has not been checked - lives in `doc/VALIDATION.md`. Accuracy figures appear
here only where they are needed to interpret a performance number.

Two summary statements set the frame for everything below.

- The coupled solver converges in far fewer outer iterations than SIMPLE
  (7.8x to 200x on the laminar cases here), needs no under-relaxation factors,
  and its outer-iteration count does not grow with mesh refinement or with the
  number of MPI ranks.
- Each outer iteration is much more expensive. At production residual levels
  on cases a well-tuned segregated solver already handles, it is currently
  slower in wall clock - 2.2x slower on the largest laminar case measured, and
  substantially slower on small RANS cases.

Contents:

1. [Method](#1-method)
2. [Laminar 2-D and small 3-D cases](#2-laminar-2-d-and-small-3-d-cases)
3. [Mesh-independence of outer
   iterations](#3-mesh-independence-of-outer-iterations)
4. [Three-dimensional wall-resolved unit
   cell](#4-three-dimensional-wall-resolved-unit-cell-483k-cells-19m-unknowns)
5. [Configuration study](#5-configuration-study)
6. [GPU](#6-gpu)
7. [RANS](#7-rans)
8. [Summary and interpretation](#8-summary-and-interpretation)
9. [Reproducing these results](#9-reproducing-these-results)


## 1. Method

**Reference solver.** Foundation OpenFOAM 14 has no standalone `simpleFoam`,
so the segregated reference throughout is `foamRun -solver incompressibleFluid`
driven by SIMPLE dictionaries, with the usual under-relaxation factors: U 0.7
and p 0.3, plus k 0.7 and epsilon 0.7 where turbulence is on. "Coupled" is
`coupledSimpleFoam` with no relaxation factors at all.

**Identical case files.** Both solvers read byte-identical `0/`, `constant/`
and `system/` directories. The coupled runs additionally read the `COUPLED`
sub-dictionary in `system/fvSolution`, which `foamRun` ignores as an unused
entry. No mesh, scheme or boundary condition differs between the two columns
of any table.

**Identical discretisation and stopping criteria.** Unless a row says
otherwise: `ddtSchemes { default steadyState; }`, `Gauss upwind` convection
(`bounded Gauss upwind` for RANS), `Gauss linear corrected` diffusion, and the
same `residualControl` for both solvers - U 1e-5 and p 1e-4 for laminar cases,
plus k 1e-4 and epsilon 1e-4 for RANS. "Iterations" means outer (Picard)
iterations for the coupled solver and SIMPLE iterations for the segregated
one; both are counted the same way, as passes over the same case.

**Timing.** Wall clock is `ExecutionTime` from the solver log, on an otherwise
idle machine unless the row is marked. Two classes of row are marked
explicitly and must not be compared against unmarked rows: rows measured while
an unrelated job shared the machine (inflated by roughly 1.5x in Section 5 and
roughly 3.5x in Section 7), and one row that is an estimate rather than a
measurement.

**Hardware and software.** A 24-core workstation running OpenFOAM 14. Eigen
3.4.0 for the serial direct and iterative backends; PETSc 3.19.6 from
distribution packages for the CPU rows; PETSc 3.25.4 built from source with
CUDA and hypre for the GPU rows; system OpenMPI 4.1.6, which is not CUDA-aware.
Rank counts are stated per row (serial, np4, np6, np8). Decomposition is
`scotch` for the two-dimensional cases and `simple` with `n (N 1 1)` for the
three-dimensional unit cell. GPU rows additionally use one workstation-class
NVIDIA card (Ada generation, 20 GB, sm_89, 48 SMs, 1/64-rate FP64) under WSL2
with no MPS available; Section 6 states the scope of those results.

**Reported quantities.** `dP` is a kinematic pressure drop (pressure divided
by density), taken from the same `functionObject` in both solvers - the
area-averaged inlet pressure, or the inlet-to-outlet difference of it.
Residual depths are the solver's own initial residuals, which use the
OpenFOAM normalisation.

**Measurement hygiene.** Repeated identical runs scatter by a few per cent in
wall clock (for example 3.80 s and 3.83 s for the same 128x128 cavity
configuration in two different tables below), and by more where preconditioner
setup dominates the iteration. Segregated iteration counts are also more
sensitive to dictionary details than coupled ones: two separately configured
runs of the 64x64 cavity at the same stated `residualControl` gave 797 and 755
SIMPLE iterations, while the coupled count was 12 in both. Where two numbers
for nominally the same thing exist, both are given rather than one being
picked.


## 2. Laminar 2-D and small 3-D cases

### 2.1 Serial suite

Five laminar cases, serial, coupled solver on the Eigen `SparseLU` backend.

| Case | Cells | Segregated iters / wall | Coupled iters / wall | Iter ratio | Wall ratio | dP segregated -> coupled |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| Lid-driven cavity 64x64, Re 100 | 4,096 | 797 / 1.57 s | 12 / 0.60 s | 66x | 2.6x | - |
| Lid-driven cavity 128x128, Re 100 | 16,384 | 2,400 / 17.5 s | 12 / 3.80 s | 200x | 4.6x | - |
| Backward-facing step, laminar | 12,225 | 843 / 5.84 s | 22 / 4.51 s | 38x | 1.3x | 43.90 -> 43.74 (0.37 %) |
| stripChannel, Re 400 | 10,920 | 208 / 1.00 s | 19 / 2.29 s | 11x | 0.44x | 1.3209 -> 1.3186 (0.17 %) |
| stripChannel, Re 1000 | 10,920 | 195 / 0.84 s | 25 / 3.13 s | 7.8x | 0.27x | 0.7959 -> 0.7928 (0.38 %) |

`stripChannel` is a two-dimensional periodic interrupted-fin channel. A wall
ratio below 1 means the coupled solver was slower.

The pattern is consistent across the suite: the iteration advantage is largest
where SIMPLE struggles most (the closed cavity, where the segregated solver
needs hundreds to thousands of iterations) and smallest where SIMPLE is
already well conditioned (the through-flow strip channel, which SIMPLE clears
in about 200 iterations). On those two strip-channel cases the coupled solver
loses on wall clock outright.

### 2.2 Small three-dimensional case

A roughly 12,000-cell duct with a half-span fin and `symmetry` side walls:

| Configuration | Iterations | Wall |
| --- | ---: | ---: |
| Coupled, Eigen `SparseLU`, serial | 16-17 | 74 s |
| Coupled, PETSc, np4 | 16-17 (identical to serial) | 2.5 s |
| Segregated | 72 | not recorded |

This is the case that establishes the practical rule that the Eigen direct
backend is a two-dimensional and debugging tool only: fill-in makes a sparse
direct factorisation uneconomical in three dimensions even at 12k cells, where
the PETSc backend at four ranks does the same work in 2.5 s. PETSc is the
three-dimensional path.

### 2.3 Accuracy

Converged results agree with the segregated solver to the level the
segregated reference is itself converged to. The summary below is what is
needed to read the tables; the full treatment is in `doc/VALIDATION.md`.

| Comparison (lid-driven cavity 64x64) | Centreline max abs(dUx) | All-cell max abs(dUx) |
| --- | ---: | ---: |
| Coupled vs segregated stopped at U 1e-5 / p 1e-4 (755 SIMPLE its) | 8.82e-4 | 1.40e-2 (lid corner cell) |
| Coupled vs segregated driven to U 1e-9 / p 1e-9 (2,021 SIMPLE its) | 1.03e-5 | 1.40e-2 (lid corner cell) |
| Coupled with `rhieChowFactor 0.7` vs the tight reference | 1.43e-6 | 9.60e-4 |

The first row is the more commonly quoted comparison and the less meaningful
one. SIMPLE residuals on this case decay at roughly 0.993 per iteration near
convergence, so stopping it at U 1e-5 still leaves of order 1e-3 of solution
error; the 8.82e-4 difference is mostly the reference's own remaining error.
Against a properly converged reference the difference is 1.03e-5. The same
exercise on the 128x128 cavity, with a reference driven to roughly 6,000
SIMPLE iterations, gives a centreline difference of 2.1e-5.

The all-cell 1.40e-2 sits in the singular lid corner cell and has a known
cause. Segregated SIMPLE takes `rAU = 1/UEqn.A()` from the *relaxed* momentum
matrix, so its converged Rhie-Chow damping is effectively `alpha_U * V/aP`. A
coupled solver has no relaxation factor, so `D_f = V/aP` is the natural
choice and is the shipped default. The two therefore differ by an O(h^2)
Rhie-Chow term that is only visible where `rAU` jumps hardest between
neighbours. Setting `rhieChowFactor 0.7` reproduces the segregated damping and
collapses the all-cell difference to 9.60e-4 and the centreline difference to
1.43e-6, which confirms the discretisations are otherwise identical. The same
mechanism, and the same fix, accounts for the residual dP differences on the
RANS backward-facing step (0.678 % at 1.0, 0.075 % at 0.7) and on the
three-dimensional unit cell (1.5 % at 1.0, 0.11 % at 0.85).

Discrete continuity is satisfied to machine precision at every outer
iteration, not merely at convergence, because the face fluxes are rebuilt from
exactly the expression that was assembled. On the 64x64 cavity the assembly
check reports `sum|div(phi)| = 1.28e-16` and `max|div(phi)| = 1.64e-18` at the
first iteration and stays there; the requirement was 1e-10 of a characteristic
flux.

### 2.4 Cross-check against an independent implementation

`tools/reference_prototype/coupled_vs_simple.py` is a separate
two-dimensional finite-volume code written from the literature, containing its
own segregated and coupled solvers on the same interrupted-fin channel
geometry. At Re 400 it predicts segregated and coupled dP of 1.3215 and
1.3198; this solver gives 1.3209 and 1.3186. Two independently written
codebases, four solutions, agreeing to about 0.3 %.


## 3. Mesh-independence of outer iterations

This is the solver's headline property, and it holds in two independent
directions: refining the mesh does not increase the outer-iteration count, and
neither does decomposing the case over more ranks.

### 3.1 Refinement

| Case | Cells | Segregated iterations | Coupled outer iterations |
| --- | ---: | ---: | ---: |
| Lid-driven cavity 64x64 | 4,096 | 797 | 12 |
| Lid-driven cavity 128x128 | 16,384 | 2,400 | 12 |
| Lid-driven cavity 256x256 | 65,536 | not measured | 12 |

A 16-fold increase in cell count leaves the coupled count at exactly 12, while
the segregated count triples over the first refinement step alone. The
independent prototype reproduces the same signature on a different geometry
and a different code base (Section 3.4).

Against a stricter segregated reference on the 64x64 cavity:

| Solver | `residualControl` | Outer iterations |
| --- | --- | ---: |
| Coupled (Eigen `SparseLU`) | U 1e-5, p 1e-4 | 12 |
| Segregated SIMPLE (relax U 0.7 / p 0.3) | U 1e-5, p 1e-4 | 755 |
| Segregated SIMPLE, tightened | U 1e-9, p 1e-9 | 2,021 |

Tightening the criterion by four orders of magnitude costs the segregated
solver 2.7x more iterations. The coupled solver reaching the same depth is
covered in Section 4.

### 3.2 Rank independence

Outer-iteration counts are identical, not within plus or minus one, between
the serial Eigen direct solve and the parallel PETSc backend:

| Case | Cells | Unknowns | Serial Eigen | np4 | np6 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Lid-driven cavity 64x64 | 4,096 | 16,384 | 12 | 12 | - |
| Lid-driven cavity 128x128 | 16,384 | 65,536 | 12 | 12 | 12 |
| Lid-driven cavity 256x256 | 65,536 | 262,144 | 12 | 12 | 12 |
| stripChannel Re 1000 | 10,920 | 43,680 | 25 | 25 | 25 |
| Backward-facing step, laminar | 12,225 | - | 22 | 22 | - |

The solutions agree to the level of the ASCII write precision, which is also
the empirical proof that the processor-patch coupling and the remote
global-index exchange are correct - a sign error there produces a
qualitatively different solution, not a 1e-10 difference:

| Comparison | Centreline max abs(dUx) | All-cell max abs(dU) |
| --- | ---: | ---: |
| Cavity 64x64, np4 vs serial Eigen `SparseLU` | 9.00e-12 | 1.21e-11 |
| Cavity 128x128, np4 vs serial Eigen `SparseLU` | 3.70e-11 | 5.71e-11 |
| Cavity 128x128, np6 vs serial Eigen `SparseLU` | 3.40e-11 | 5.01e-11 |
| Cavity 256x256, np4 vs serial Eigen `SparseLU` | 2.05e-10 | 2.09e-10 |
| Cavity 256x256, np6 vs serial Eigen `SparseLU` | 2.03e-10 | 2.07e-10 |
| Cavity 64x64, serial PETSc vs serial Eigen | 9.00e-12 | 9.55e-12 |

The requirement was 1e-8. The floor is partly the `writePrecision 12` ASCII
output, that is about 1e-12 on values of order one. On the strip-channel case
the parallel dP reproduces the serial value to all printed digits
(7.92849560e-01) at both np4 and np6.

### 3.3 What that costs in wall clock

Iteration counts being flat does not make wall clock flat. Laminar cavities,
PETSc rows using FGMRES with the fieldsplit-Schur preconditioner:

| Case | Serial Eigen `SparseLU` | Serial PETSc | np4 | np6 |
| --- | ---: | ---: | ---: | ---: |
| Cavity 64x64 (16k unknowns) | 0.60 s | 1.04 s | 0.38 s | - |
| Cavity 128x128 (66k unknowns) | 3.83 s | - | 2.96 s | 2.15 s |
| Cavity 256x256 (262k unknowns) | 37.7 s | 122.9 s | 50.1 s | 41.8 s |

For two-dimensional problems of this size a sparse direct solve is hard to
beat: Eigen `SparseLU` is still the fastest option at 262k unknowns. The PETSc
path scales sensibly with rank count (122.9 s to 50.1 s to 41.8 s, that is
2.45x on four ranks at 61 % efficiency and 2.94x on six at 49 %) and is the
only option once a direct factorisation no longer fits in memory or the mesh
is three-dimensional. The efficiency loss is growth of the fieldsplit
iteration count with the number of subdomains, not communication.

Further parallel wall-clock points at unchanged iteration counts: strip
channel Re 1000, 3.13 s serial / 2.57 s np4 / 1.79 s np6, all 25 iterations;
backward-facing step laminar, 4.51 s serial / 2.35 s np4, both 22 iterations.

### 3.4 Independent prototype

The same experiment in `tools/reference_prototype`, an independently written
two-dimensional finite-volume code with the same discretisation (upwind
convection, central diffusion, Rhie-Chow fluxes, direct LU inner solve) on an
interrupted-fin channel:

| Grid, Re | SIMPLE (0.7 / 0.3) | Coupled (no relaxation) | Iter ratio | dP agreement |
| --- | ---: | ---: | ---: | ---: |
| 120x24, Re 400 | 149 | 29 | 5.1x | 0.17 % |
| 240x48, Re 400 | 284 | 27 | 10.5x | 0.13 % |
| 240x48, Re 1000 | 241 | 31 | 7.8x | 0.26 % |
| Any grid, relaxation 0.85 / 0.5 | diverged | not applicable (no relaxation used) | - | - |

SIMPLE iterations grew from 149 to 284 under refinement while the coupled
count stayed at 27-31, and SIMPLE diverged outright when the relaxation
factors were pushed to 0.85 / 0.5, where the coupled solver has no relaxation
factors to get wrong. Wall clock in that code was 2.8-3.7x in the coupled
solver's favour even with a naive direct LU inner solve.


## 4. Three-dimensional wall-resolved unit cell (483k cells, 1.9M unknowns)

Every row in this section is the same laminar case and the same mesh: a
three-dimensional interrupted-fin unit cell, 483k cells generated with
`snappyHexMesh` (maximum non-orthogonality 80.9 deg, 5,408 skew faces),
1.93M unknowns, decomposed with `simple`, `n (N 1 1)`, at np6. The segregated
reference is `foamRun -solver incompressibleFluid` at np6: 142 iterations,
41 s.

### 4.1 Headline comparison at production criteria

| | Segregated (np6) | Coupled (fieldsplit-Schur + hypre, np6) |
| --- | ---: | ---: |
| Outer iterations | 142 | 24 |
| Wall clock | 41 s | 91-95 s |
| dP (kinematic) | 670.44 | 660.25 at the default `rhieChowFactor 1.0`; 669.68 at 0.85 (0.11 %) |
| Reaches 1e-7 residuals | no - 8,000 iterations, 2,334 s, never reached | yes, at 28 outer iterations |

At production accuracy the segregated solver is 2.2x faster in wall clock
while taking 5.9x more iterations. The coupled solver is the only one of the
two that can be driven to deep residuals on this mesh at all.

### 4.2 Progression: what each change bought

Same case, same mesh, np6 throughout. Each row differs from the one above it
by the single change named.

| Configuration | Outer iters | Wall | s/outer | Residual depth (U / p) | dP | What the change bought |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Segregated reference | 142 | 41 s | - | ~1e-6 floor | 670.44 | - |
| Coupled baseline: fieldsplit-Schur + GAMG, inner `relTol 1e-12` | 20 | 496 s | ~25 | 5e-7 / 3e-6 | 660.25 | robust, but pays for near-exact inner solves |
| + pseudo-transient continuation, inner `relTol 1e-6` | 21 | 259 s | 12.3 | 1.9e-7 / 2.3e-6 | 660.49 | 1.9x - looser inner solves at the same outer count |
| + batched COO assembly | 21 | 257.3 s | 12.2 | - | - | 0.7 % - the insertion loop was never the bottleneck |
| + hypre BoomerAMG instead of GAMG on the Schur block | 24 | 91-95 s | 3.7-3.8 | - | 660.49 | 2.7x - inner iterations 188 -> ~75, one options-string line, no GPU |

The last row is the important one. It is a one-line change to `petscOptions`,
requires no rebuild and no accelerator, and is worth more than every other
optimisation in this document combined. It is the shipped default.

Batched assembly is included because it is a measured negative:
`MatSetPreallocationCOO` / `MatSetValuesCOO` replaced roughly 30 million
individual `MatSetValues` calls with one call per iteration, and bought 0.1 s
per iteration. The per-iteration cost is the inner Krylov solve plus
preconditioner setup, and nothing else. The change is numerically equivalent -
on the strip-channel case at np4 the batched path reproduces both the
iteration count and the pressure drop to every printed digit - so the residual
and dP columns of that row are not separately reported. (The same batched API is
PETSc's device assembly API, which is what the GPU path uses; see Section 6.)

A re-measurement of the GAMG-on-Schur configuration at a different time gave
21 outer iterations in 221.4 s (9.77 s/outer) rather than 257 s (12.2 s/outer).
Both are reported; the difference is machine state, not configuration, and it
is the reason the tables prefer s/outer and ratios to absolute seconds.

### 4.3 Deep convergence

Driving both solvers to 1e-7 rather than production criteria:

| Run | Iterations | Wall | Residual reached | dP |
| --- | ---: | ---: | --- | ---: |
| Segregated, target 1e-7 | 8,000 (cap) | 2,334 s | never reached | - |
| Coupled, cold start, `rhieChowFactor 0.85` | 28 | ~690 s (estimated solo, not a clean measurement) | 8e-10 / 2.2e-8 | 669.68 |

The segregated solver does not reach the target at any cost on this mesh: it
runs to its 8,000-iteration cap in 39 minutes and stops short. The coupled
solver reaches three orders deeper than production criteria for eight extra
outer iterations over the 20 it needs for production. The wall-clock figure in
that row is an estimate of what the run would have taken alone on the machine,
not a measurement, and is marked accordingly.

### 4.4 Answer independence and mass conservation

Every converged coupled configuration on this case lands within 0.04 % of
every other: 660.25 to 660.51 at `rhieChowFactor 1.0`, and 669.67 to 669.68 at
0.85. The answer does not depend on the preconditioner, the inner tolerance,
the rank count or the backend; only speed and robustness do.

The remaining gap to the segregated solver on this mesh is accounted for. The
first run of this case converged in 36 iterations but produced a dP 9.6 % low,
because `symmetry` patches do not report `fixesValue()` and their
per-component transform coefficients cannot represent exactly zero mass flux on
tilted `snappyHexMesh` faces. Treating impermeable constraint patches
(symmetry, symmetryPlane, wedge, slip) as zero-flux in the continuity row
closed that to 1.5 %, and matching the Rhie-Chow damping definition with
`rhieChowFactor 0.85` closes the rest to 0.11 % - that is, the entire
remainder is the documented damping definition of Section 2.3 and not a
discretisation difference.


## 5. Configuration study

Alternatives that were measured and rejected, reported because a negative
result is worth as much to a reader as a positive one.

### 5.1 Strategies on the 483k-cell unit cell (np6)

Rows marked (c) shared the machine with an unrelated job: their ratios are
meaningful, their absolute times are inflated by roughly 1.5x and must not be
compared with unmarked rows.

| Option | Outer iters | Wall | Residual depth | dP | Outcome |
| --- | ---: | ---: | --- | ---: | --- |
| Pseudo-transient + inner `relTol 1e-4` | 60 (cap) | ~308 s (c) | 6e-6 / 1.0e-4 | 660.51 | too loose - the extra outer iterations eat the gain |
| No damping + inner `relTol 1e-4` (control) | 80 | 185 s | plateaus at ~6e-6 / 1e-4 | 660.27 | cheapest per iteration, but stalls; damping is what converts loose inner solves into convergence |
| Multiplicative fieldsplit (SIMPLE-preconditioner variant) | 20 | ~807 s (c) | 1.6e-7 | 660.2535 | same family, same answer as Schur, no gain |
| GAMG applied to the whole block system | failed | killed at 20 min for 1 iteration | - | - | plain scalar AMG cannot handle the stabilised saddle-point block system |
| Warm start: converge segregated, then hand the field to the coupled solver and drive to 1e-7 | 142 seg + 21 coupled | 41 + 686 = 727 s | 1.4e-9 / 4.5e-8 | 669.67 | negative: saves about 7 coupled iterations. The outer contraction rate, not the starting distance, sets the count |
| `reusePreconditioner yes` (pseudo-transient, inner 1e-6) | 21 | 392.7 s vs 257.3 s fresh | as baseline | as baseline | counterproductive, +53 %; freezing only the hierarchy structure costs +66 % |
| Batched COO assembly (pseudo-transient, inner 1e-6) | 21 | 257.3 s vs 259.0 s | as baseline | as baseline | 0.7 %, within noise of nothing |

The preconditioner-reuse result has a design consequence: the Picard operator
changes fast enough between outer iterations that a stale preconditioner costs
more inner iterations than its setup saved. Any future block-AMG work has to
make fresh setup intrinsically cheap rather than amortise setup across
iterations.

### 5.2 Preconditioner choice depends on mesh topology

This is why there is no obviously correct default, and why the default that
did ship was arrived at by measurement.

| Configuration | Cavity-class 2-D meshes | 2-D interrupted-fin strip channel | 3-D `snappyHexMesh` unit cell (1.93M unknowns) |
| --- | --- | --- | --- |
| ASM (overlap 2) / sub-ILU(2) | converges, 78.9 s at 262k unknowns np6 | converges everywhere tested, 2.92 s np4 | caps out - 5,000 inner iterations, ~440 s/outer, unusable |
| fieldsplit-Schur (selfp) + GAMG | converges, 42.2 s at 262k unknowns np6 | **diverges**, serial and parallel - `DIVERGED_ITS` at 5,000 inner iterations | converges, 130-470 inner iterations, ~12.2 s/outer |
| fieldsplit-Schur (selfp) + hypre BoomerAMG | converges | converges, and to the serial direct reference's every printed digit | converges, ~75 inner iterations, 3.7-3.8 s/outer |
| hypre BoomerAMG on the whole system (scalar) | - | diverges at 5,000 inner iterations | - |
| hypre BoomerAMG nodal (systems AMG) on the whole system | - | converges to the reference (2,361 inner iterations, 11.9 s) | healthy inner solves, but did not close the outer loop under the uniform pseudo-transient recipe; per-inner cost is ASM-class |
| GAMG on the whole block system | - | - | total failure - killed at 20 min for 1 iteration |
| LSC | - | converges (21 outer iterations) | 211 s/outer - about 40x the practical target |
| Upper-Schur, unsmoothed GAMG + ILU | - | inner solve diverges | - |
| Vanka-style `PCPATCH` | - | needs a `DM`; unusable from this assembly | - |

ASM/ILU and fieldsplit-Schur + GAMG are mirror images: each is robust exactly
where the other fails. Only fieldsplit-Schur + hypre BoomerAMG passes both,
which is why it is the shipped default. A preconditioner default validated on
one mesh topology is validated on that topology alone: the strip-channel
divergence does not appear on cavity-class meshes at all.

A direct comparison on one mesh, cavity 256x256 at np6 with `relTol 1e-12`:

| Preconditioner | Wall | Outer iterations |
| --- | ---: | ---: |
| ASM (overlap 2) / sub-ILU(2) | 78.9 s | 12 |
| fieldsplit-Schur (selfp) + GAMG | 42.2 s | 12 |

The same contrast on the 128x128 cavity at np6, measured separately: 6.29 s
with ASM/ILU against 2.17 s with fieldsplit.

### 5.3 Inner tolerance is regime-dependent and cannot be set once

A loose inner tolerance does not merely slow the outer loop, it stops it
converging: the outer loop is a Picard iteration with a nonzero initial guess,
so an inner solve that barely moves the iterate produces an outer update that
barely moves the solution. On the laminar cavity, `relTol 1e-3` gave "did NOT
converge" in 4 s and `relTol 1e-6` the same in 13 s, while `relTol 1e-12`
converges.

| Regime | Recommended `relTol` | Evidence |
| --- | --- | --- |
| Small laminar, 2-D | 1e-12 | 1e-3 and 1e-6 both fail to converge on the cavity |
| Large laminar 3-D with pseudo-transient continuation | 1e-6 | 1.9x faster than 1e-12 at the same outer count; 1e-4 costs more outer iterations than it saves, and without damping it plateaus outright |
| Small RANS | 1e-12 (do not loosen) | at 1e-6 the summed absolute divergence holds only to 1.8e-9 (normalised 2.8e-8) instead of machine precision, and the outer loop does not finish within 1,000 iterations |
| Large 3-D RANS | 1e-8 with `maxLinearIter 500`, plus pseudo-transient | at 1e-12 that matrix needs ~2,340 inner iterations per outer solve, about 168 s/outer - intractable |

The large-3-D-RANS row is the exception to the "do not loosen for RANS" rule
and exists because `nuEff` heterogeneity degrades both the velocity-block
preconditioner and the `selfp` Schur approximation at once.


## 6. GPU

**Scope.** Every GPU number here was measured on one workstation-class NVIDIA
card (Ada generation, 20 GB, sm_89, 48 SMs, 1/64-rate FP64) under WSL2, with
no MPS available and a system OpenMPI that is not CUDA-aware. Under WSL2 each
MPI rank creates its own CUDA context on the single device (about 0.4 s and
300 MB per rank) and the contexts time-slice. These conclusions are specific
to that hardware and platform class and should not be read as statements about
GPU acceleration in general.

**Enabling the device path** costs nothing in the solver: adding
`-mat_type aijcusparse -vec_type cuda` to `petscOptions` is sufficient, with no
rebuild of the solver logic and no change on the OpenFOAM side, because the
batched COO assembly adopted for MPI is already the PETSc GPU assembly API.

### 6.1 Correctness

The CPU sanity gate, run with the CUDA-enabled PETSc but no GPU flags, on the
strip channel at Re 1000, np4, ASM(2)/ILU(2), `relTol 1e-12`:

| Build | Outer iterations | Inlet dP |
| --- | ---: | --- |
| Reference: serial Eigen `SparseLU` | 25 | 7.92849560e-01 |
| Distribution PETSc 3.19.6, np4 | 25 | 7.92849560e-01 |
| Source PETSc 3.25.4 with CUDA, np4 | 25 | 7.92849560e-01 |

The device path is equally exact. Because the outer loop is a Picard iteration
driven to a fixed `residualControl`, any preconditioner that genuinely solves
the linear system to 1e-12 must reproduce the reference, which makes this small
case a correctness gate rather than a smoke test:

| Configuration (43,680 unknowns, np4) | Outer | dP | Wall | Last inner KSP its |
| --- | ---: | --- | ---: | --- |
| CPU ASM/ILU(2) | 25 | 7.92849560e-01 | 2.92 s | 645 |
| GPU ASM/ILU(2) | 25 | 7.92849560e-01 | 98.6 s | 561 |
| CPU bjacobi/ILU | 25 | 7.92849560e-01 | 4.77 s | 1418 |
| GPU bjacobi/ILU | 25 | 7.92849560e-01 | 133.3 s | 1416 |
| CPU jacobi | 25 | 7.92849560e-01 | 19.8 s | 5000 (`DIVERGED_ITS`) |
| GPU jacobi | 21 of 30 | - | timeout at 180 s | 5000 (`DIVERGED_ITS`) |
| CPU hypre BoomerAMG (scalar, whole system) | 30, no convergence | - | 12.7 s | 5000 (`DIVERGED_ITS`) |
| GPU hypre BoomerAMG (scalar, whole system) | 5 of 30 | - | timeout at 180 s | 858 (converged) |
| CPU hypre BoomerAMG nodal | 25 | 7.92849560e-01 | 11.9 s | 2361 |
| GPU hypre BoomerAMG nodal | - | - | segmentation fault | - |
| CPU fieldsplit-Schur + GAMG | 30, no convergence | - | 95.0 s | 5000 (`DIVERGED_ITS`) |
| GPU fieldsplit-Schur + GAMG | 2 of 30 | - | timeout at 180 s | 5000 (`DIVERGED_ITS`) |
| GPU AmgX CLASSICAL | - | - | segmentation fault | - |
| GPU AmgX AGGREGATION | - | - | segmentation fault | - |

Both converging preconditioners reproduce the reference exactly on the device.
At 43,680 unknowns the GPU is about 30x slower, which is the expected result:
roughly 11k unknowns per rank is far below the kernel-launch break-even. Note
also that hypre BoomerAMG behaves differently on CPU and GPU by design,
because hypre's device build must use device-compatible coarsening and
interpolation (PMIS with extended interpolation) where the CPU default is
Falgout with classical interpolation.

### 6.2 Screening on the 483k-cell unit cell

np6, five outer iterations, ranked by steady s/outer = (T5 - T1) / 4, which
excludes one-off CUDA context creation and first preconditioner setup. `FS` is
fieldsplit Schur, `fact_type full`, `precondition selfp`, split 0,1,2 / 3.

| Configuration | Velocity block | Schur block | s/outer | Inner KSP its | GPU util (max) |
| --- | --- | --- | ---: | ---: | ---: |
| CPU FS + GAMG (the then-default) | bjacobi/ILU | GAMG | 19.95 | 188 | 5 % |
| GPU FS + GAMG | bjacobi/ILU | GAMG | 16.52 | 188 | 99 % |
| GPU FS + hypre | bjacobi/ILU | BoomerAMG | 8.25 | 73 | 98 % |
| GPU FS + hypre, cuSPARSE ILU | bjacobi/ILU-cusparse | BoomerAMG | 8.24 | 73 | 98 % |
| GPU FS + hypre, Jacobi velocity block | jacobi | BoomerAMG | 15.75 | 181 | 97 % |
| CPU FS + hypre | bjacobi/ILU | BoomerAMG | 7.72 | 77 | 19 % |
| GPU ASM/ILU-cusparse (monolithic) | - | - | 36.32 | 2000 (`DIVERGED`) | 99 % |

Four readings. The large win is algorithmic, not architectural: GAMG to hypre
on the Schur block is 2.6x, entirely on the CPU. With the preconditioner held
fixed the GPU loses by 7 % (8.25 against 7.72). The GPU only wins where the CPU
preconditioner is weak (16.52 against 19.95 with GAMG, 1.21x), which is a
bigger device share of a worse algorithm. And this is not an under-utilisation
problem: the device sits at 97-99 % utilisation and 7-8 GB of 20 GB throughout.
Two further negatives: replacing the velocity-block ILU sub-solve with a
cuSPARSE ILU, which removes a host fallback, is worth 0.01 s/outer, and
replacing that block with plain Jacobi to make it fully device-resident nearly
doubles the inner iteration count. Removing host fallbacks is not where the
time is.

### 6.3 Full runs and rank scaling

483k-cell unit cell, pseudo-transient, `pseudoCo 200`, `relTol 1e-6`,
`maxLinearIter 2000`.

| Run | np | Outer | Wall | s/outer | Inlet dP | GPU util max/mean | GPU memory |
| --- | ---: | ---: | ---: | ---: | --- | --- | ---: |
| GPU FS + hypre | 4 | 25 | 94.3 s | 3.55 | 6.60489958e+02 | 98 % / 81 % | 9.1 GB |
| GPU FS + hypre | 6 | 26 | 116.4 s | 4.28 | 6.60489962e+02 | 98 % / 87 % | 10.2 GB |
| GPU FS + hypre | 8 | 25 | 156.7 s | 5.90 | 6.60489711e+02 | 98 % / 87 % | 10.1 GB |
| CPU FS + hypre | 4 | 30 | 119.9 s | 3.69 | 6.60489721e+02 | - | - |
| CPU FS + hypre | 6 | 24 | 94.7 s | 3.73 | 6.60489519e+02 | - | - |
| CPU FS + hypre | 8 | 30 | 104.7 s | 3.23 | 6.60490012e+02 | - | - |
| CPU FS + GAMG (re-measured) | 6 | 21 | 221.4 s | 9.77 | 6.60489298e+02 | - | - |
| CPU FS + GAMG (original measurement) | 6 | 21 | 257 s | 12.2 | 6.60489172e+02 | - | - |
| Segregated reference | 6 | 142 | 41 s | - | - | - | - |

GPU scaling is inverted - 3.55, 4.28, 5.90 s/outer at 4, 6 and 8 ranks. Every
rank added makes it slower, because without MPS each rank opens its own CUDA
context on one device and the contexts time-slice, and more subdomains also
means a slightly weaker preconditioner. The CPU control over the same range is
flat to improving (3.69, 3.73, 3.23). On this hardware the GPU wants the fewest
ranks that still fit and the CPU wants the most.

All configurations agree on the answer to 6-7 significant digits against the
CPU FS+GAMG value 6.60489172e+02: GPU np4 1.2e-6, GPU np6 1.2e-6, GPU np8
8.2e-7, CPU FS+hypre np6 5.3e-7. The spread is the expected consequence of
`relTol 1e-6` stopping the outer loop at 24-30 rather than 21 iterations with a
different preconditioner, not a discretisation difference; the bit-level check
is Section 6.1, where the tight `relTol 1e-12` runs reproduce the serial
reference to all printed digits on CPU and GPU alike.

### 6.4 Where the time goes, and why the speedup is modest

`-log_view` profile, np6, best GPU configuration, 117.4 s wall, 26 outer
iterations, 2.284e11 total flops:

| Event | % of wall | % of flops | GPU %F | Host to device | Device to host |
| --- | ---: | ---: | ---: | --- | --- |
| `KSPSolve` | 58 % | 98 % | 100 | 3559 calls / 645 MB | 3372 / 170 MB |
| `PCApply` | 50 % | 29 % | 100 | 1736 / 429 MB | 1686 / 83 MB |
| `KSPSolve_FS_Schur` (hypre on the pressure block) | 44 % | 0 %* | 0* | - | - |
| `PCSetUpOnBlocks` (velocity ILU numeric factorisation) | 25 % | 1 % | 100 | 125 / 1.57 GB | - |
| `PCSetUp` | 10 % | 2 % | 92 | 244 / 1.61 GB | 224 / 1.82 GB |
| `MatMult` | 9 % | 40 % | 100 | 2648 / 580 MB | 2572 / 170 MB |
| `MatMatMultSym` (selfp Schur build) | 7 % | 2 % | 100 | 200 / 1.54 GB | 75 / 220 MB |
| `MatSetPreallocationCOO` | 4 % | 0 % | - | 102 / 1.30 GB | - |
| `MatSetValuesCOO` (the assembly handoff) | 1 % | 0 % | - | 0 | 0 |
| `MatCUSPARSECopyTo` | 2 % | - | - | 352 / 3.50 GB | - |
| `MatCUSPARSECopyFrom` | 0 % | - | - | - | 100 / 1.53 GB |

\* hypre performs its own device work internally and PETSc cannot attribute
those flops, so the column reads zero while the wall clock is real.

Three things follow. First, the assembly handoff is free: `MatSetValuesCOO` is
1 % of wall with zero logged transfers, so the OpenFOAM-side assembly is not a
bottleneck and needed no GPU-specific work. Second, about 70 % of wall clock is
preconditioner - 44 % is the single AMG solve on the Schur block and another
25 % is velocity-block ILU setup - so the achievable device speedup is bounded
by how much of the preconditioner runs on the device, not by the Krylov
kernels. Third, 1.53 GB is copied back device to host per run, because
`PCFIELDSPLIT` extracts submatrices and hypre needs its own IJ format, and both
round-trip through the host every outer iteration.

That last point is the structural reason the device version cannot win here:
the matrix is rebuilt and re-shipped every Picard step, so there is nothing to
amortise. The arithmetic reinforces it. The aggregate rate over `KSPSolve` is
3.29 GFlop/s. Sparse double-precision linear algebra is bandwidth-bound rather
than flop-bound - a sparse matrix-vector product does about two flops per
matrix entry loaded - so the useful figure of merit is memory bandwidth, not
peak FP64. On top of that this is a 1/64-rate FP64 workstation part, so the
dense-arithmetic advantage that a datacentre card would bring is absent as
well. A modest result on this hardware is what the hardware predicts.

### 6.5 Verdict, and the library-level blockers

| | Outer | Wall (best) | vs CPU coupled 257 s | vs segregated 41 s |
| --- | ---: | ---: | --- | --- |
| Segregated (np6) | 142 | 41 s | - | 1.0x |
| CPU coupled, FS + GAMG (previous default, np6) | 21 | 257 s | 1.0x | 6.3x slower |
| CPU coupled, FS + hypre (np6) | 24 | 94.7 s | 2.7x faster | 2.3x slower |
| GPU coupled, FS + hypre (np4) | 25 | 94.3 s | 2.7x faster | 2.3x slower |

The GPU ties the CPU: 94.3 s against 94.7 s is a 0.4 % difference. At fixed
rank count and fixed preconditioner the GPU is ahead only at np4 and loses at
np6 and np8, and the best s/outer in the entire study is a CPU point (3.23
s/outer at np8). The 2.7x came from the algorithm change, not the hardware.

Two library-level routes to a device-resident *block* AMG - which is what this
system actually wants, since the unknowns are genuinely block-4 interleaved -
were tried and are closed on this platform:

- **AmgX 2.4.0** (the version PETSc pins) hits three walls. It links
  `CUDA::nvToolsExt`, which CUDA 12.9 removed, so the toolkit must be pinned to
  12.8. PETSc's `PCSetUp_AMGX` then hands AmgX a mixed host/device CSR - device
  values from `MatSeqAIJCUSPARSEGetArrayRead` with host indices from
  `MatGetRowIJ`, which `MATSEQAIJCUSPARSE` does not override - and the upload
  fails in Thrust. Both were worked around locally. The third wall is not
  workaroundable: with everything else fixed,
  `AMGX_matrix_upload_distributed` segfaults inside the WSL2 virtualised CUDA
  driver's asynchronous host-to-device copy. On PETSc's own
  `snes/tutorials/ex19` (block size 4, the same structure as this system)
  `-pc_type amgx` works at np1 on a ~4.4k-row problem and segfaults at ~263k
  and ~1.05M rows; at more than one rank it fails for any matrix and vector
  type, including all-host `aij`, in AmgX's distributed consolidation path;
  and the aggregation path reports `Unsupported device/host algorithm` for
  every selector, because AmgX resolves those to its host selector
  implementations, none of which are implemented on the host.
  The two PETSc source changes used to get that far were made
  locally and have not been submitted or confirmed against current PETSc; they
  should be treated as local workarounds, not as fixed upstream bugs. A working
  AmgX would need bare-metal Linux, or a newer AmgX than the pinned 2.4.0.
- **hypre's nodal systems AMG has no device implementation** and segfaults on
  the GPU. It is the only AMG measured to reproduce the reference on the hard
  two-dimensional topology, so this closes the second route.

What remains available on the device is scalar AMG on the Schur complement,
which is exactly the shipped configuration.

Based on these measurements rather than hope, a GPU would plausibly pay off
with: bare-metal Linux and a working block-AMG library; MPS, or one rank per
GPU, since rank count actively hurts here; a datacentre part with 1/2-rate
rather than 1/64-rate FP64; and a mesh large enough that the per-Picard matrix
re-ship stops dominating. None of those applied to the runs above.


## 7. RANS

Turbulence is solved segregated and lagged one outer iteration behind the
coupled pressure-velocity solve, using OpenFOAM's own `momentumTransport`
models. Only `kEpsilon` and `kOmegaSST` have been exercised; other
incompressible RAS and LES models are selectable through the run-time
selection tables of the linked libraries but are untested.

### 7.1 Backward-facing step, kEpsilon, 12,225 cells

Stock tutorial `0/` and `constant/` (`simulationType RAS`, `model kEpsilon`,
`turbulence on`, `viscosityModel constant`, `nu 1e-05`) with a steady-ised
`system/`. `residualControl` is the same for both solvers: p 1e-4, U 1e-5,
k 1e-4, epsilon 1e-4.

Every wall-clock figure in this table is inflated by roughly 3.5x by an
unrelated concurrent job on the same host. Iteration counts and pressures are
unaffected; the times may be compared with each other but not with any other
table in this document.

| Solver | Ranks | Iterations | ExecutionTime (contended) | Inlet area-averaged p |
| --- | ---: | --- | ---: | --- |
| Coupled, Eigen `SparseLU` | 1 | 779 (converged) | 389 s | -5.05255551608 |
| Coupled, PETSc, `relTol 1e-12` (default) | 4 | 771 (converged) | 1629 s | -5.05287256579 |
| Coupled, PETSc, `relTol 1e-6` | 4 | 1000, not converged (`Uy` 5.95e-05; every other criterion met) | 321 s | -5.05250882903 |
| Coupled, Eigen `SparseLU`, `rhieChowFactor 0.7` | 1 | 762 (converged) | 621 s | -5.01474242228 |
| Segregated SIMPLE | 1 | 1,997 (converged) | 61 s | -5.02168382241 |
| Segregated SIMPLE | 4 | 2,453 (converged) | 33 s | -5.01900246749 |
| Segregated, tightened (p 1e-6, U 1e-7, k/epsilon 1e-6) | 1 | 30,000 cap; p floors at 1.8e-04 | 977 s | -5.0183 (band -5.01828 to -5.01851) |

Serial and np4 agree: 779 against 771 outer iterations (1 %) and inlet
pressures matching to 0.006 %. The counts are not bit-identical the way the
laminar cases are, because at these tolerances the PETSc path's
non-reproducible reductions move the final iteration by a few.

The tightened segregated reference never meets p 1e-6 - SIMPLE's own p
residual floors at 1.8e-04 on this case, with `Ux` at 1.3e-06 and k and epsilon
at 8.8e-06 - so it runs to its 30,000-step cap and its inlet pressure wanders
over about +/-0.005 %. That matters when reading the pressure column: the
difference between the coupled and segregated pressures is of the same order
as the segregated reference's own uncertainty, and most of the remainder is the
Rhie-Chow damping definition of Section 2.3 (the `rhieChowFactor 0.7` row
reduces the dP gap from 0.678 % to 0.075 %, a 9x reduction at essentially the
same iteration count).

Correctness gates on this case are all met: the RAS model is selected and both
`k` and `epsilon` solve every outer iteration with monotonically decreasing
initial residuals (1 to 9.8e-06 over 779 iterations); there are no `bounding k`
or `bounding epsilon` warnings, no NaN and no floating-point exceptions; and
`sum|div(phi)|` stays at machine precision throughout (2.2e-17 at iteration 1,
1.4e-16 at iteration 779), so the coupled continuity rows are still satisfied
exactly with a turbulent, strongly varying `nuEff`.

### 7.2 The lagged turbulence sets the convergence floor for both solvers

Convergence on this case has two phases. Phase 1 is fast and is the coupled
solver doing what it does: `Ux` falls from 1 to 1.7e-03 in 61 outer iterations
where SIMPLE needs 451. Phase 2 is a long plateau at `Ux` around 5e-05 and p
around 1.7e-04 that lasts hundreds of iterations and then collapses by two
orders in the last few iterations, once k and epsilon reach about 1e-05.

Four diagnostic runs, identical except for the one setting named:

| Variant | Change | Iterations (cap) | `Ux` | `p` | `k` |
| --- | --- | --- | --- | --- | --- |
| V0 | baseline, `pseudoCo 200` | 200 (cap) | 2.54e-04 | 2.44e-04 | 7.74e-04 |
| V1 | `pseudoTransient no` | 400 (cap) | 1.47e-04 | 1.43e-04 | 1.10e-04 |
| V2 | `pseudoCo 50` | 400 (cap) | 4.73e-05 | 2.02e-04 | 1.19e-04 |
| V3 | `div(phi,U) bounded Gauss upwind` | 400 (cap) | 5.21e-05 | 1.44e-04 | 1.02e-04 |
| V4 | baseline, cap 1000 | 779 (converged) | 2.20e-06 | 1.16e-06 | 9.84e-06 |

No setting removes the plateau, because the plateau is not a property of the
pressure-velocity coupling. Restarting V2's iteration-400 fields with
`turbulence off`, so that `nut` is frozen and nothing else changes, makes the
coupled block converge geometrically at about 0.82 per iteration: from
`Ux` 3.95e-05 and p 1.15e-04 at iteration 401 to `Ux` 2.87e-08 and p 7.20e-09
at iteration 440. Forty iterations cover three to four orders.

The mechanism is straightforward. The coupled residual is
`b_{n+1} - A_{n+1} x_n`, and `A_{n+1}` contains `D_f = V/aP` with `aP` built
from `nuEff = nu + nut`. While `nut` changes, the operator itself changes
between iterations, and the residual cannot vanish however well the
pressure-velocity block is solved. The same floor appears in the segregated
solver, so this is a property of lagged RANS rather than of this solver.

Against SIMPLE at matched residual levels the coupled solver keeps a 4-7x
per-iteration advantage:

| Target | Coupled outer its | SIMPLE its |
| --- | ---: | ---: |
| `Ux` < 1.7e-03 | 61 | 451 |
| `Ux` < 3.2e-04 | 181 | 777 |
| `k` < 9.5e-04 | 181 | 773 |

End to end it is 779 against 1,997 iterations, 2.6x fewer. But each coupled
iteration is far more expensive, so on this 12k-cell two-dimensional RANS case
SIMPLE wins on wall clock, 61 s against 389 s under the same contended
conditions.

### 7.3 Cost of the turbulence-capable momentum operator

The one clean timing comparison in this section: the momentum operator built
through `momentumTransport->divDevSigma(U)` against the earlier hand-written
laminar operator, both binaries timed back to back under identical machine
load, best of three runs each.

| Case | New operator | Old operator | Ratio |
| --- | ---: | ---: | ---: |
| Lid-driven cavity 64x64, 12 iterations | 2.147 s | 2.036 s | 1.055 |
| stripChannel Re 1000, 25 iterations | 11.166 s | 11.083 s | 1.008 |

That is 1-5 %, from the extra `fvVectorMatrix` temporaries the framework
operator builds. Both columns are about 3.5x the idle-machine values elsewhere
in this document, which is exactly why the ratio and not the absolute time is
the meaningful number here.

### 7.4 Other RANS cases

Strip channel, kEpsilon at Re 1e5, an independently built case: segregated 224
iterations at dP 0.27639; coupled 148 iterations, converged, zero bounding
events, dP 0.26939 - 2.5 % apart.

Three-dimensional interrupted-fin unit cell, kEpsilon at Re 2e4, 483k cells,
np6. Both solvers hit the lag floor on this marginal case (wall functions on a
low-y+ `snappyHexMesh` mesh), so the comparison is what each achieves in a
comparable compute budget:

| | Segregated | Coupled (`relTol 1e-8`, inner cap 500) |
| --- | --- | --- |
| Iterations / wall | 8,000 (cap) / 3,456 s | 150 (cap) / 3,760 s |
| U residual floor | 1.5e-2 | 3.2e-3 (4.7x deeper) |
| p residual floor | 4.9e-4 | 8.8e-4 (comparable) |
| dP (kinematic) | 300.43 | 291.32 (3.0 % apart; both floors wander) |
| k-bounding events | 11 | 4 |

At the shipped tight tolerance the turbulent version of this matrix needs about
2,340 inner iterations per outer solve (roughly 168 s/outer, intractable),
because `nuEff` heterogeneity degrades both the velocity-block preconditioner
and the `selfp` Schur approximation. The measured recipe for this class of case
is `relTol 1e-8` with `maxLinearIter 500` and pseudo-transient continuation.

### 7.5 kOmegaSST, and a regime caveat that applies to both solvers

`kOmegaSST` additionally requires `wallDist { method meshWave; }` in
`system/fvSchemes`. On the two-dimensional strip channel at genuinely
laminar-transitional conditions the coupled solver takes 146 iterations
serially and 140 at four ranks (residuals to 5e-7) against 195 for the
segregated solver, with zero bounding events and dP within 0.38 %. These are
the values `tutorials/stripChannelRAS` reproduces. Both solvers relaminarise to the laminar answers
to three digits (0.793-0.796 against the laminar 0.7928-0.7959), which shows
that SST does no harm where the flow is genuinely laminar.

Wall-resolved SST at transitional Reynolds numbers on three-dimensional
`snappyHexMesh` interrupted-fin meshes is pathological for *both* solvers:

| Leg | Outcome |
| --- | --- |
| Re 1000, segregated | 6,000-iteration cap, 3,245 bounding events, dP 22,221 - 33x the laminar value, that is, garbage |
| Re 2000, segregated | 6,000-iteration cap, 5,530 bounding events, dP 1,787, an unconverged limit cycle |
| Re 1000, coupled | the same pathology - residuals pinned at 0.86 with bounding through 116 capped iterations |
| Re 2000, coupled | not run; the evidence is complete without it |

This is a modelling-regime statement, not a solver defect. At Re 500 to 2,000
these flows are laminar-transitional and laminar modelling is the correct
physics. SST is available and behaves correctly in genuinely turbulent regimes
(demonstrated at Re 2e4) and relaminarises harmlessly on clean meshes.

### 7.6 RANS verdict

The turbulence hook is correct - laminar cases are preserved bit for bit, the
independently built RANS cases converge to the expected answers, and the
Rhie-Chow damping attribution reproduces in RANS exactly as in laminar flow -
and the coupled solver retains a large iteration advantage. But the lagged
turbulence erases the wall-clock case at these scales: per-iteration costs rise
sharply and the lag sets the convergence floor for both solvers. For production
RANS the segregated solver remains the workhorse. The coupled solver's RANS
niche is a modestly deeper residual floor at equal compute. Pulling k and
epsilon into the block system is a documented future direction, not implemented.


## 8. Summary and interpretation

### 8.1 Where this solver wins

- **Outer-iteration count.** 7.8x to 200x fewer iterations than SIMPLE on the
  laminar cases here, 5.9x on the 483k-cell three-dimensional case, 2.6x end to
  end on RANS and 4-7x at matched residual levels.
- **No relaxation tuning.** There are no pressure-velocity relaxation factors
  to choose, and therefore none to get wrong. The independent prototype
  diverges as SIMPLE at 0.85 / 0.5 relaxation on a mesh where the coupled
  algorithm converges in 27 iterations with no tuning at all.
- **Deep convergence.** On the 483k-cell mesh the segregated solver cannot
  reach 1e-7 at any cost (8,000 iterations, 39 minutes, target never reached),
  while the coupled solver gets there in 28 outer iterations - eight more than
  it needs for production criteria.
- **Mesh and rank independence.** 12 outer iterations at 4k, 16k and 65k cells;
  identical counts serial, np4 and np6 on every case tested, with solutions
  agreeing to 1e-10 or better.
- **The answer does not depend on the solver settings.** Every converged
  configuration on the three-dimensional case agrees within 0.04 %, across
  preconditioners, inner tolerances, rank counts, CPU and GPU.

### 8.2 Where it loses

- **Wall clock at production tolerances**, on cases a segregated solver already
  handles well. 2.2x behind on the 483k-cell unit cell (91-95 s against 41 s),
  and a straight loss on the small two-dimensional through-flow cases (wall
  ratios 0.44x and 0.27x).
- **RANS wall clock.** 389 s against 61 s on the 12k-cell RANS backward-facing
  step (both contended, same conditions), and a dead heat at best on the
  three-dimensional RANS case, where both solvers are lag-limited anyway.
- **Per-case tuning moves from relaxation factors to the linear solver.**
  `relTol` cannot be set once and forgotten: 1e-12 for small laminar, 1e-6 for
  large laminar with pseudo-transient continuation, 1e-12 for small RANS, 1e-8
  with a capped `maxLinearIter` for large RANS.
- **No universal preconditioner.** ASM/ILU and Schur+GAMG each fail where the
  other works; only Schur+hypre passed every topology tested, and it is the
  default precisely for that reason.
- **No GPU speedup on workstation-class hardware.** The device path is exact
  and fully functional but ties the CPU at best on the hardware measured.
- **Memory.** The block system is four times the unknowns of any single
  segregated equation, and the Eigen direct backend is unusable in three
  dimensions.

### 8.3 When should I use this solver?

Reach for it when:

- the segregated solver stalls, oscillates, or needs aggressive
  under-relaxation to stay alive on the case;
- the results must be converged much more deeply than SIMPLE's residual floor,
  for example for adjoint or optimisation work, or for verification runs where
  the discretisation error must be isolated from iterative error;
- iteration count matters more than seconds - a mesh-refinement or
  parameter-sweep study where the segregated iteration count grows with
  refinement and the coupled one does not;
- the case must run without per-case relaxation tuning, for instance inside an
  automated sweep with no human in the loop.

Stay with the segregated solver when:

- the case is a well-behaved laminar or RANS problem that SIMPLE already
  converges in a few hundred iterations, and total wall clock is the objective;
- the case is RANS at production accuracy - the lagged turbulence caps what the
  coupled pressure-velocity solve can buy;
- the case needs anything outside this solver's scope: transient, compressible,
  multiphase, energy, `fvModels`, `fvConstraints` or MRF.


## 9. Reproducing these results

`./Allrun` at the repository root builds nothing and runs every tutorial,
printing a pass/fail summary; each tutorial can also be run on its own from its
own directory. The tutorial directories are named for the geometries used in
the tables above.

| Rows | Case | Where it comes from |
| --- | --- | --- |
| Sections 2.1, 2.3, 3.1-3.3 cavity 64x64 rows | Lid-driven cavity 64x64, Re 100 | `tutorials/cavity` |
| Section 3 cavity 128x128 and 256x256 rows | Lid-driven cavity, refined | `tutorials/cavity` with the `blockMeshDict` cell counts doubled or quadrupled |
| stripChannel Re 400 and Re 1000 | 2-D periodic interrupted-fin channel | `tutorials/stripChannel` (Re 1000 as shipped; Re 400 by raising `nu` in `constant/physicalProperties`) |
| Section 2.2 three-dimensional duct rows | Interrupted-fin duct, ~12k cells | `tutorials/finnedDuct` |
| Section 7.5 kOmegaSST rows | 2-D interrupted-fin channel, kOmegaSST | `tutorials/stripChannelRAS` |
| Section 2.4 and 3.4 prototype rows | Independent 2-D finite-volume code | `tools/reference_prototype/coupled_vs_simple.py`; needs Python 3 with NumPy and SciPy, no OpenFOAM |
| Backward-facing step rows (laminar, Sections 7.1-7.3 kEpsilon) | Backward-facing step | **Not shipped** - reproducible from the stock OpenFOAM `pitzDaily` `blockMesh` resource, see below |
| Sections 4, 6, 7.4 rows | 483k-cell interrupted-fin unit cell | **Not distributable** - see below |

**What is not in the repository.** The 483k-cell unit-cell mesh is a
`snappyHexMesh` result of several hundred megabytes and is not distributable,
so Sections 4, 6 and 7.4 are reported as measurements rather than as
reproducible tutorials. The backward-facing step is not shipped either: its
mesh comes from the stock OpenFOAM `pitzDaily` `blockMesh` resource, which any
OpenFOAM installation already provides, and the case is a standard steady
incompressible setup on that mesh. Everything those sections claim about the solver's
behaviour that can be shown on a distributable case - iteration counts
independent of mesh size and rank count, preconditioner-independent answers,
the effect of the inner tolerance, the Rhie-Chow damping attribution - is also
shown on the two-dimensional tutorials.

The GPU rows additionally require a CUDA-enabled PETSc build (see
`doc/INSTALL.md`) and are specific to the hardware and platform described in
Section 6.

**Running a comparison.** Both solvers read the same case directory, so a
comparison is two runs and a diff:

```sh
cd tutorials/<case>
./Allrun                                    # coupled run
foamRun -solver incompressibleFluid         # segregated reference, same case
```

`tools/compare_fields.py` compares the velocity fields of two case and time
directories cell by cell on a structured two-dimensional mesh, which is how the
coupled solution is checked against a segregated reference, a parallel run
against a serial one, or one linear-algebra backend against another.
`doc/VALIDATION.md` gives the full re-verification procedure, including the
comparisons that are not performance measurements.

**What should reproduce, and what should not.** Outer-iteration counts on the
Eigen `SparseLU` path are deterministic and should reproduce exactly, as should
converged pressure drops to the digits printed here. PETSc runs use
non-reproducible parallel reductions, so their iteration counts can differ by a
few at tight tolerances even on the same machine, and their results agree to
roughly seven digits rather than bitwise. Wall-clock times will not reproduce
on different hardware; the ratios between rows of the same table are the
transferable quantity.
