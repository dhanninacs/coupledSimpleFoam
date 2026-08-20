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
- Each outer iteration is much more expensive, so wall clock turns on problem
  size. The crossover on this hardware sits near 300k cells (Section 9.4):
  below it the segregated solver is ahead, and on the small two-dimensional
  through-flow cases it wins outright; above it the flat coupled iteration
  count compounds, and with the tuned configuration of Section 9.2 the coupled
  solver is 1.3x faster at 533k cells, 2.0x at 1.07M and 2.3x at 2.15M. RANS is
  the exception at every size measured, because the lagged turbulence sets the
  convergence floor for both solvers.

This document is chronological, and the three-dimensional sections were written
as the configuration improved. Where an early section records a configuration
that was slower than segregated at a size where the tuned one is now faster,
the numbers stand as they were measured and the text says which section
supersedes the conclusion.

Contents:

1. [Method](#1-method)
2. [Laminar 2-D and small 3-D cases](#2-laminar-2-d-and-small-3-d-cases)
3. [Mesh-independence of outer
   iterations](#3-mesh-independence-of-outer-iterations)
4. [Three-dimensional wall-resolved unit
   cell](#4-three-dimensional-wall-resolved-unit-cell-483k-cells-19m-unknowns)
5. [Configuration study](#5-configuration-study)
6. [RANS](#6-rans)
7. [Summary and interpretation](#7-summary-and-interpretation)
8. [Scale-up: 2.15 million cells, 8.6 million
   unknowns](#8-scale-up-215-million-cells-86-million-unknowns)
9. [Beating the segregated solver: the tuned large-case
   configuration](#9-beating-the-segregated-solver-the-tuned-large-case-configuration)
10. [Reproducing these results](#10-reproducing-these-results)


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
roughly 3.5x in Section 6), and one row that is an estimate rather than a
measurement.

**Measurement protocol.** Every timing in this document is a single run with the
machine otherwise idle. Later measurements were serialised with an advisory file
lock, because an earlier harness that polled for a free machine allowed two
batches to overlap; where that affected a number it has been re-measured and the
re-measured value is the one quoted. Repeats within one session agree to about
**2 %** (a 2.15M-cell coupled run repeated back to back at 218 s and 214 s).
Independent sessions differ by up to
about **5 %**: the 2.15M-cell pair quoted throughout as 203 s coupled against
478 s segregated was re-measured under the lock at 218 s and 214 s against 488 s. The
ratio is stable at **2.3 times** in both, and ratios rather than absolute
seconds are what should be carried away from this document. Differences below
5 % between two configurations measured in different sessions are not
meaningful.

**Hardware and software.** A 24-core workstation running OpenFOAM 14. Eigen
3.4.0 for the serial direct and iterative backends; PETSc with hypre for the
PETSc rows, 3.19.6 from distribution packages and 3.25.4 built from source;
system OpenMPI 4.1.6. Rank counts are stated per row. Decomposition is
`scotch` for the two-dimensional cases and `simple` with `n (N 1 1)` for the
three-dimensional unit cell.

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

That 2.2x is a property of the configuration this section measures, not of the
size. It predates the adaptive inner tolerance, the periodic preconditioner
rebuild and the three-dimensional AMG settings of Section 9, each of which is
worth a large factor at and above this scale; on the structured meshes of
Section 9.4 the tuned configuration is already ahead by 1.3x at 533k cells.
The tuned configuration has not been re-run on this `snappyHexMesh` mesh, so
every row here stands exactly as measured and none of them should be read as
the current wall-clock verdict.

### 4.2 Progression: what each change bought

Same case, same mesh, np6 throughout. Each row differs from the one above it
by the single change named.

| Configuration | Outer iters | Wall | s/outer | Residual depth (U / p) | dP | What the change bought |
| --- | ---: | ---: | ---: | --- | ---: | --- |
| Segregated reference | 142 | 41 s | - | ~1e-6 floor | 670.44 | - |
| Coupled baseline: fieldsplit-Schur + GAMG, inner `relTol 1e-12` | 20 | 496 s | ~25 | 5e-7 / 3e-6 | 660.25 | robust, but pays for near-exact inner solves |
| + pseudo-transient continuation, inner `relTol 1e-6` | 21 | 259 s | 12.3 | 1.9e-7 / 2.3e-6 | 660.49 | 1.9x - looser inner solves at the same outer count |
| + batched COO assembly | 21 | 257.3 s | 12.2 | - | - | 0.7 % - the insertion loop was never the bottleneck |
| + hypre BoomerAMG instead of GAMG on the Schur block | 24 | 91-95 s | 3.7-3.8 | - | 660.49 | 2.7x - inner iterations 188 -> ~75, one options-string line |

The last row is the important one. It is a one-line change to `petscOptions`,
requires no rebuild and no accelerator, and at 2.7x is still the largest single
factor any one change has bought in this document. It is the shipped default.
The four changes of Section 9.1 are each smaller than it and compound to more
than it; two of them are options-string lines like this one, and two set solver
options added since.

Batched assembly is included because it is a measured negative:
`MatSetPreallocationCOO` / `MatSetValuesCOO` replaced roughly 30 million
individual `MatSetValues` calls with one call per iteration, and bought 0.1 s
per iteration. The per-iteration cost is the inner Krylov solve plus
preconditioner setup, and nothing else. The change is numerically equivalent -
on the strip-channel case at np4 the batched path reproduces both the
iteration count and the pressure drop to every printed digit - so the residual
and dP columns of that row are not separately reported.

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

The preconditioner-reuse result reads as a design consequence: the Picard
operator changes fast enough between outer iterations that a stale
preconditioner costs more inner iterations than its setup saved. Two later
measurements narrow that statement. The row above freezes the hierarchy for the
whole run, whereas rebuilding it on a fixed interval is worth 6 % at 8.6M
unknowns and, once each cycle had been made cheap by the three-dimensional AMG
settings, a factor of 1.7 at the same size (Section 9.1); the useful rule is
that reuse pays in proportion to how cheap each application already is, not
that reuse never pays. And the block-AMG alternative is no longer an open
question: it was built through hypre and PETSc, measured at 8.6M unknowns and
rejected on cost (Section 9.3).

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

Size matters as well as topology, and it begins to matter above the sizes in
this table. The `selfp` Schur approximation forms an explicit sparse matrix
product every outer iteration; at 1.93M unknowns that is affordable, and at
8.6M unknowns it becomes the dominant cost and is replaced by the `a11` option,
which reuses the pressure block the assembly already builds (Section 8.1).
`selfp` remains the shipped default
because it is the more robust of the two on small and strongly non-orthogonal
meshes, which is the majority of what the solver is pointed at.

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

Every row above is a fixed tolerance, and for the large laminar
three-dimensional regime the choice has since been superseded rather than
corrected: `adaptiveTolerance` derives the inner tolerance from the ratio of
successive nonlinear block residuals instead, which is both faster than any
fixed value tried and, at 8.6M unknowns, the difference between converging and
not (Section 9.1). The rows above still describe the fixed-tolerance path,
which is what the shipped defaults use.


## 6. RANS

Turbulence is solved segregated and lagged one outer iteration behind the
coupled pressure-velocity solve, using OpenFOAM's own `momentumTransport`
models. Only `kEpsilon` and `kOmegaSST` have been exercised; other
incompressible RAS and LES models are selectable through the run-time
selection tables of the linked libraries but are untested.

### 6.1 Backward-facing step, kEpsilon, 12,225 cells

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

### 6.2 The lagged turbulence sets the convergence floor for both solvers

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

### 6.3 Cost of the turbulence-capable momentum operator

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

### 6.4 Other RANS cases

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

### 6.5 kOmegaSST, and a regime caveat that applies to both solvers

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

### 6.6 RANS verdict

The turbulence hook is correct - laminar cases are preserved bit for bit, the
independently built RANS cases converge to the expected answers, and the
Rhie-Chow damping attribution reproduces in RANS exactly as in laminar flow -
and the coupled solver retains a large iteration advantage. But the lagged
turbulence erases the wall-clock case at these scales: per-iteration costs rise
sharply and the lag sets the convergence floor for both solvers. For production
RANS the segregated solver remains the workhorse. The coupled solver's RANS
niche is a modestly deeper residual floor at equal compute. Pulling k and
epsilon into the block system is a documented future direction, not implemented.


## 7. Summary and interpretation

### 7.1 Where this solver wins

- **Wall clock on large three-dimensional laminar cases.** With the tuned
  configuration of Section 9.2 the coupled solver is 1.3x faster at 533k
  cells, 2.0x at 1.07M and 2.3x at 2.15M, on identical meshes, identical
  schemes and identical convergence criteria, with both solvers at their best
  settings. The margin grows with the mesh because only one of the two has an
  iteration count that grows with it.
- **Outer-iteration count.** 7.8x to 200x fewer iterations than SIMPLE on the
  laminar cases here, 5.9x on the 483k-cell three-dimensional case, 19x on the
  2.15M-cell one (555 against 29), 2.6x end to end on RANS and 4-7x at matched
  residual levels.
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
  preconditioners, inner tolerances, rank counts and backends.

### 7.2 Where it loses

- **Wall clock below the crossover**, on small cases a segregated solver
  already handles well: a straight loss on the two-dimensional through-flow
  cases (wall ratios 0.44x and 0.27x) and no better than a tie at 269k cells
  (0.93x). The 483k-cell unit cell is recorded at 2.2x behind (91-95 s against
  41 s) with the configuration of Section 4, which predates the tuning of
  Section 9 and has not been re-measured with it.
- **RANS wall clock.** 389 s against 61 s on the 12k-cell RANS backward-facing
  step (both contended, same conditions), and a dead heat at best on the
  three-dimensional RANS case, where both solvers are lag-limited anyway.
- **Per-case tuning moves from relaxation factors to the linear solver.** A
  fixed `relTol` cannot be set once and forgotten: 1e-12 for small laminar,
  1e-6 for large laminar with pseudo-transient continuation, 1e-12 for small
  RANS, 1e-8 with a capped `maxLinearIter` for large RANS. `adaptiveTolerance`
  removes that choice for the large three-dimensional laminar case, where it
  also outperforms every fixed value tried, but the shipped configuration for
  such a case is still a page of `petscOptions` (Section 9.2) rather than a
  default.
- **No universal preconditioner.** ASM/ILU and Schur+GAMG each fail where the
  other works; only Schur+hypre passed every topology tested, and it is the
  default precisely for that reason.
- **Memory.** The block system is four times the unknowns of any single
  segregated equation, and the Eigen direct backend is unusable in three
  dimensions.

### 7.3 When should I use this solver?

Reach for it when:

- the case is a three-dimensional laminar mesh of roughly 300k cells or more
  and wall clock is the objective - that is where the crossover sits, and above
  a million cells the tuned configuration of Section 9.2 wins by a factor of
  two or more;
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

- the case is small - two-dimensional, or three-dimensional below a few hundred
  thousand cells - and SIMPLE already converges it in a few hundred iterations,
  and total wall clock is the objective;
- the case is RANS at production accuracy - the lagged turbulence caps what the
  coupled pressure-velocity solve can buy;
- the case needs anything outside this solver's scope: transient, compressible,
  multiphase, energy, `fvModels`, `fvConstraints` or MRF.


## 8. Scale-up: 2.15 million cells, 8.6 million unknowns

A structured three-dimensional interrupted-fin duct, 2 150 400 hexahedral cells
(400 x 100 x 56 with four staggered fins carved out), laminar at Re ~ 500,
symmetry side walls, zero mesh non-orthogonality. Ten MPI ranks for both
solvers, identical mesh, schemes and `residualControl` (`p` 1e-4, `U` 1e-5).
The coupled system carries 8 601 600 unknowns, 4.5 times the case of Section 4.
The rows below are the first measurement at this size, with the configuration as
it stood then; Section 9 revisits the same case and the same mesh with a tuned
one and supersedes the wall-clock conclusion drawn here.

| | Segregated | Coupled (`a11` Schur, pseudo-transient, `relTol` 1e-6) |
| --- | --- | --- |
| Outer iterations | 555 (formal convergence) | 19-20 to the same streamwise and pressure targets |
| Wall clock | 480 s | 1112 s at iteration 20; 1236 s for the full 60-iteration run |
| Final residuals | U 5.4e-6, p 5.6e-6 | Ux 6.7e-7, p 2.0e-7; Uy 1.3e-5, Uz 1.1e-5 |
| dP (kinematic) | 0.96269858 | 0.96256354 (**0.014 %** apart) |
| sum abs div(phi) | - | 1.0e-7 (1.2e-9 normalised) |
| Peak resident memory | - | 37.5 GB across ten ranks |

Field agreement over all 2 150 400 cells: max cell-wise velocity difference
5.15e-2 against a field maximum of 1.68 (3.1 %, localised at fin corners), rms
difference 1.28e-3, that is 0.076 % of the field maximum. The localisation and
the size ratio between max and rms are the signature of the Rhie-Chow damping
difference of Section 2.3, not of a discretisation disagreement.

**Iteration independence holds at scale.** Refining from the 483k-cell case of
Section 4 to this one, a factor 4.5 in cells, moved the segregated count from
142 to 555 (a factor 3.9) while the coupled count moved from 24 to 20. The
coupled outer count is set by the nonlinearity, not by the mesh.

**Wall-clock ratio, as this configuration left it.** The coupled solver was 2.2
times slower than segregated at 483k cells and is 2.3 times slower here, and at
the time that pair of numbers looked like a scaling law: the cost of one coupled
iteration appeared to grow faster than linearly with problem size, the Krylov
iteration count per outer solve rising with the system and cancelling the
growing iteration-count advantage. It was a tuning state and not a scaling law.
The 1112 s in the table above is an untuned baseline - a fixed inner tolerance,
a full preconditioner rebuild every outer iteration, restarted FGMRES, and a
Schur-block AMG left on its two-dimensional defaults - and each of those is
worth a large factor at this size. Section 9 takes the same case and the same
mesh to 203 s against the segregated solver's 478 s, and Section 9.4 places the
crossover near 300k cells rather than beyond reach. The measurements in this
section stand; the conclusion drawn from them does not.

**Cost profile.** Almost all the time is spent early: iterations 1-10 cost 762 s
of the 1236 s total, iteration 19 is reached at 1095 s, and the marginal cost of
iterations 30-60 is 1.5 s each, because the solution has stopped changing and
each inner solve converges immediately.

**Transverse components set the stopping criterion.** `Ux` and `p` fall 20-50
times below their targets while `Uy` and `Uz` plateau just above `1e-5` (1.3e-5
and 1.1e-5). In a strongly directional flow the transverse residual
normalisation divides by a near-zero reference response, so those components
stall at a floor that has no bearing on the solution: the fields are frozen to
four significant figures from iteration 30 onwards. Judge such a case on the
streamwise component and the pressure, or set a looser tolerance for the
transverse components. The tuned configuration of Section 9 does meet `U` 1e-5
on all three components on this case, so the plateau recorded here belongs to
the stalled inner-solve configuration above rather than to the geometry; the
normalisation argument still explains why the transverse components are the
last to fall.

### 8.1 Preconditioner choice at this size

Screening at 8.6 million unknowns, three outer iterations each:

| `-pc_fieldsplit_schur_precondition` / preconditioner | s per outer iteration |
| --- | --- |
| `a11` + BoomerAMG (own pressure block) | **98.7** |
| ASM(1) / ILU(1) | 112.3 |
| `selfp` + BoomerAMG, `relTol` 1e-3 | 278 |
| `selfp` + BoomerAMG, `relTol` 1e-6 (shipped default) | 555 |

The shipped `selfp` default forms an explicit sparse matrix product for the
Schur approximation and rebuilds it every outer iteration; at this size that
product dominates the solve. The `a11` option uses the solver's own implicit
pressure-Laplacian block instead, which costs nothing to form and, on this
mesh, converges the same. All four reach the same residual after three
iterations, so the difference is cost rather than convergence quality.


## 9. Beating the segregated solver: the tuned large-case configuration

On the 2 150 400-cell case of Section 8 (8 601 600 unknowns, ten MPI ranks) a
tuned configuration converges in **203 s against the segregated solver's 478 s**
- 2.3 times faster - with the same accuracy. Both numbers are reproducible: the
coupled run was repeated back to back at 203 s / 203 s, and the segregated
reference at 480 s / 478 s.

| | Segregated | Coupled, tuned |
| --- | --- | --- |
| Outer iterations | 555 | **29** |
| Wall clock | 478 s | **203 s** |
| Converged | yes | yes |
| dP | 0.96269619 | 0.96255897 (0.014 %) |
| Velocity field, all 2.15M cells | - | rms difference 1.26e-3 = 0.075 % of the field maximum |

The field difference is the same as the one produced by a slow, tightly solved
run (1.28e-3), so the speed comes from the linear algebra and not from a
looser answer. The normalised `sum|div(phi)|` is about 1e-9, as in Section 8.

The two solver settings this configuration relies on are both off by default -
`adaptiveTolerance no`, and `pcReuseInterval 1`, which rebuilds the
preconditioner every outer iteration - so nothing in Sections 2 to 9 is
affected by their existence and every number in them stands as measured.

### 9.1 How the factor of 5.5 was obtained

Starting from the Section 8 configuration at 1112 s, four independent changes
compound. Each was measured separately on the same case:

| Change | Wall clock | Why |
| --- | --- | --- |
| Section 8 baseline (`a11`, fixed `relTol` 1e-6) | 1112 s | did not satisfy the convergence test at all |
| `adaptiveTolerance yes`, `relTolMax` 1e-2 | 710 s | stops over-solving early iterations; **also fixes the convergence stall** |
| `pcReuseInterval 3` | 665 s | the AMG hierarchy no longer rebuilt every outer iteration |
| `-ksp_type fbcgs` instead of restarted FGMRES | 556 s | removes orthogonalisation against a 30-vector basis of 8.6M unknowns |
| 3-D tuning of the Schur-block AMG | 354 s | see below - the single largest lever |
| `pcReuseInterval 8` | **203 s** | cheap V-cycles make a longer reuse interval pay |

**Why the adaptive tolerance also fixes the stall.** The forcing term is an
Eisenstat-Walker ratio of successive nonlinear block residuals, clamped to
`relTolMin` and `relTolMax`, and it is defined against the current iterate, so
the adaptive path additionally switches the inner convergence test to the
initial-residual norm. That switch is not cosmetic: under PETSc's default test,
which is relative to the norm of the right-hand side, a warm-started loose solve
satisfies its tolerance before it takes a single Krylov step, and the outer loop
stops moving. The fixed-`relTol` path is unchanged and still uses the default
test.

**The Schur-block AMG was running on two-dimensional defaults.** PETSc's
BoomerAMG default `strong_threshold` is 0.25; hypre's own guidance for
three-dimensional problems is 0.5, and the lower value marks far too many
connections as strong, inflating operator complexity and the cost of every
V-cycle. Correcting that, with cheaper HMIS coarsening and truncated `ext+i`
interpolation, cut the run from 556 s to 354 s. Once each cycle was cheap, a
longer preconditioner reuse interval became profitable and took it to 203 s.
Reuse interval 8-12 is the optimum (206 s at 12, 238 s at 16).

### 9.2 The configuration

```c
COUPLED
{
    nOuterIterations    80;
    pseudoTransient     yes;
    pseudoCo            50;

    adaptiveTolerance   yes;
    relTolMax           3e-2;
    relTolMin           1e-8;
    adaptiveGamma       0.9;
    adaptiveAlpha       1.5;

    pcReuseInterval     8;
    maxLinearIter       500;

    petscOptions
        "-ksp_type fbcgs \
         -pc_type fieldsplit -pc_fieldsplit_type schur \
         -pc_fieldsplit_schur_fact_type upper \
         -pc_fieldsplit_schur_precondition a11 \
         -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 \
         -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type bjacobi \
         -fieldsplit_0_sub_pc_type ilu \
         -fieldsplit_1_ksp_type preonly \
         -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type boomeramg \
         -fieldsplit_1_pc_hypre_boomeramg_strong_threshold 0.5 \
         -fieldsplit_1_pc_hypre_boomeramg_coarsen_type HMIS \
         -fieldsplit_1_pc_hypre_boomeramg_interp_type ext+i \
         -fieldsplit_1_pc_hypre_boomeramg_P_max 4 \
         -fieldsplit_1_pc_hypre_boomeramg_truncfactor 0.3 \
         -fieldsplit_1_pc_hypre_boomeramg_agg_nl 1 \
         -fieldsplit_1_pc_hypre_boomeramg_relax_type_all SOR/Jacobi";

    residualControl { U 1e-5; p 1e-4; }
}
```

A `petscOptions` string that spans several lines **must** end each continued
line with a backslash. OpenFOAM's dictionary parser reads the value as a single
quoted token and rejects a bare newline inside it with

```text
--> FOAM FATAL IO ERROR:
found '\n' while reading string "-ksp_type fgmre..."
```

which is easy to mistake for a solver problem rather than a formatting one.
Writing the whole string on one long line is equally valid and never fails.

Use it for three-dimensional cases of roughly a million cells and up. On small
two-dimensional cases the shipped defaults remain the better choice: the
adaptive ceiling of 3e-2 is too loose there (see doc/USAGE.md), and neither
preconditioner reuse nor the 3-D AMG settings pay when the hierarchy is cheap.

### 9.3 Monolithic block AMG: measured and rejected

Reproducing a Fluent-style monolithic coupled AMG through hypre and PETSc was
tried and does not compete on this system. All runs used
`-pc_hypre_boomeramg_numfunctions 4`, without which PETSc leaves hypre at
`num_functions = 1` and every systems-AMG option silently degenerates; every
run also carried `-options_left`, which reported no unused options.

| Preconditioner | s per outer iteration | Residual after 3 outer iterations |
| --- | --- | --- |
| Schur `a11` (adopted) | 98.7 | 0.037 |
| hypre nodal, block interpolation + nodal Schwarz smoother | 363-370 | 0.058-0.061 |
| hypre nodal systems-AMG (Frobenius) | 389 | 0.977 |
| hypre nodal systems-AMG (row sum) | 684 | 0.968 |
| GAMG, unsmoothed aggregation + block Jacobi | fails | diverges with NaN on the first application |
| Best block AMG plus diagonal equation scaling | 292 | 0.114 |

The architecture is sound - block interpolation with a block smoother improves
the three-iteration residual sixteen-fold over plain nodal coarsening on the
same matrix - but through generic library components it costs three to four
times more per iteration than the Schur splitting. A competitive monolithic
block AMG needs purpose-built coarsening, interpolation and smoothers, which is
a solver project rather than a configuration change.


### 9.4 Where the crossover sits: a four-point size sweep

The same interrupted-fin geometry meshed at four resolutions, both solvers at
their best settings, ten MPI ranks throughout, identical convergence criteria
(`U` 1e-5, `p` 1e-4). Every run formally converged.

| Cells | Segregated | Coupled (tuned, 9.2) | Speed ratio | dP agreement | Field rms |
| --- | --- | --- | --- | --- | --- |
| 269k | 221 it / 13 s | 21 it / 14 s | 0.93x (tie) | 0.18 % | 0.12 % |
| 533k | 296 it / 44 s | 24 it / 35 s | **1.3x coupled** | 0.12 % | 0.090 % |
| 1.07M | 404 it / 149 s | 25 it / 75 s | **2.0x coupled** | 0.063 % | 0.074 % |
| 2.15M | 555 it / 478 s | 29 it / 203 s | **2.3x coupled** | 0.014 % | 0.075 % |

Two clean scaling laws produce the crossover. The segregated iteration count
grows with resolution (221 -> 296 -> 404 -> 555) while the coupled count is
essentially flat (21 -> 24 -> 25 -> 29), so the coupled advantage compounds
with size; the crossover on this hardware sits near **300k cells**, and above a
million cells the tuned coupled solver wins by a factor of two or more. The
pressure-drop and field agreement *improve* with refinement (0.18 % down to
0.014 %), consistent with the O(h^2) Rhie-Chow damping attribution of
Section 2.3.


### 9.5 Heat transfer: the advantage roughly doubles

The same 2 150 400-cell duct with the energy equation active: inlet 300 K, all
walls and fins isothermal at 350 K, Pr 0.71, laminar. The segregated reference is
`foamRun` with `incompressibleFluid` plus a `scalarTransport` function object
carrying the identical discrete temperature equation (constant diffusivity
`nu/Pr`); the coupled solver uses `solveEnergy yes`. Both legs use the same
temperature linear-solver settings, both were run exclusively under a lock, and
both converged.

| | Ranks | Outer | Temperature solves | Wall clock | dP | Bulk temperature rise |
| --- | ---: | ---: | ---: | ---: | --- | ---: |
| Segregated | 10 | 555 | 555 | **1099 s** | 0.96269533 | 14.86343 K |
| **Coupled** | 10 | 29 | 28 | **245 s** | 0.96255905 | 14.86796 K |
| Coupled | 20 | 28 | 27 | 250 s | 0.96255657 | 14.86796 K |

**Coupled is 4.5 times faster than segregated with heat transfer on**, against
2.3 times for the same case without it. The mechanism is the mesh-independent
outer count: a segregated run solves the temperature equation once per SIMPLE
iteration and therefore pays for 555 of them, while the coupled run pays for 28.
Each additional transported scalar - a species, a second temperature, a tracer -
is solved once per outer iteration in both codes, so each one widens the gap
again by the same ratio.

Accuracy is unaffected. The bulk temperature rise agrees to **0.031 %**
(14.86343 K against 14.86796 K), the peak temperature to four significant
figures (349.127 K against 349.124 K), and dP to 0.014 % as in the isothermal
case. The coupled bulk rise is identical to five decimals across ten ranks and
twenty ranks.

**Do not over-solve the temperature equation.** These numbers were first
measured with `T { tolerance 1e-12; relTol 0; }`, which is the right setting for
the validation work of `doc/VALIDATION.md` and badly wrong for production: the
convergence target is `residualControl T 1e-6`, so on later outer iterations the
*initial* residual was already below target and the solver still spent 62 to 104
inner iterations driving it to 1e-13. Switching to `relTol 0.01` cuts that to 16
to 19 inner iterations, changes the answer in no digit reported above, and is
worth a factor of **2.2 on the segregated leg** (2366 s to 1099 s) and 1.3 on the
coupled one (322 s to 245 s). It costs the segregated solver more because it
repeats the waste 555 times rather than 28. Treat `T` like `p` and `U`: a loose
relative tolerance, with the outer `residualControl` deciding convergence.


### 9.6 Rank scaling, and what did not work

All rows below are single exclusive runs on the 2 150 400-cell case under the
lock described in Section 1.

**Neither solver benefits from more than about six cores.**

| Ranks | Coupled, tuned | Segregated |
| ---: | ---: | ---: |
| 6 | 213 s | - |
| 10 | 218 / 214 s | 488 s |
| 16 | 235 s | - |
| 20 | 216 s | 492 s |

The coupled solver is flat from 6 to 20 ranks and the segregated one from 10 to
20. Both saturate memory bandwidth well before ten cores on this machine, so the
coupled solver delivers its full advantage on **six cores** and the remaining
cores are better spent running further cases concurrently than adding ranks to
one case. Outer-iteration counts are unchanged across the range (28-29 coupled,
555-556 segregated), so this is a hardware limit and not an algorithmic one.

**Options-only tuning beyond the configuration of Section 9.2 produced
nothing.** A sweep of aggressive-coarsening levels, strength threshold 0.5 to
0.7, interpolation truncation `P_max` 2 to 5, hypre's l1-Gauss-Seidel smoother
in place of hybrid SOR, `KSPFBCGSR` in place of `KSPFBCGS`, and RCM renumbering
all landed within the 5 % session spread of the baseline. Two of those arms did
not test what they appeared to: `-pc_hypre_boomeramg_agg_interp_type` **is not a
PETSc option** and `-fieldsplit_0_pc_factor_mat_ordering_type` needs a `sub_`
prefix to reach a `bjacobi` sub-preconditioner. Both were reported by
`-options_left`, which is the reason to pass it on every exploratory run: a
misspelled PETSc option is silently ignored and produces a convincing null
result.


## 10. Reproducing these results

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
| Section 6.5 kOmegaSST rows | 2-D interrupted-fin channel, kOmegaSST | `tutorials/stripChannelRAS` |
| Section 2.4 and 3.4 prototype rows | Independent 2-D finite-volume code | `tools/reference_prototype/coupled_vs_simple.py`; needs Python 3 with NumPy and SciPy, no OpenFOAM |
| Backward-facing step rows (laminar, Sections 6.1-6.3 kEpsilon) | Backward-facing step | **Not shipped** - reproducible from the stock OpenFOAM `pitzDaily` `blockMesh` resource, see below |
| Sections 4, 5 and 6.4 rows | 483k-cell interrupted-fin unit cell | **Not distributable** - see below |
| Sections 8, 9 rows | Structured interrupted-fin duct, 269k to 2.15M cells | **Not shipped** - a hexahedral duct with staggered fins carved out, meshed at four resolutions, see below |

**What is not in the repository.** The 483k-cell unit-cell mesh is a
`snappyHexMesh` result of several hundred megabytes and is not distributable,
so Sections 4, 5 and 6.4 are reported as measurements rather than as
reproducible tutorials. The structured duct of Sections 8 and 9 is not
shipped for the opposite reason: it is trivially describable, a hexahedral duct
with four staggered fins carved out at 400 x 100 x 56 cells in its finest
resolution, but that resolution needs ten ranks and 37.5 GB of memory, which is
not a tutorial. The backward-facing step is not shipped either: its
mesh comes from the stock OpenFOAM `pitzDaily` `blockMesh` resource, which any
OpenFOAM installation already provides, and the case is a standard steady
incompressible setup on that mesh. Everything those sections claim about the solver's
behaviour that can be shown on a distributable case - iteration counts
independent of mesh size and rank count, preconditioner-independent answers,
the effect of the inner tolerance, the Rhie-Chow damping attribution - is also
shown on the two-dimensional tutorials.

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
