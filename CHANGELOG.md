# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## Unreleased

### Added

- `solveEnergy` (default `no`) with `Pr` and `Prt`: a temperature transport
  equation solved segregated after the coupled pressure-velocity step, on the
  conservative Rhie-Chow flux the coupled solve already produces. Verified
  against the analytical Nusselt number for fully developed laminar flow between
  isothermal plates (7.5416 computed against 7.5407 exact, 0.012 %), against a
  segregated reference solving the identical discrete equation, and by a global
  energy balance closing to roughly 1e-9 percent. Valid for constant properties
  without buoyancy, where temperature is a passive scalar.
- `tutorials/heatedDuct`, covered by the top-level `Allrun`.

### Documented

- Heat transfer roughly doubles the coupled solver's advantage, to 4.5 times on
  a 2.15M-cell case, because a segregated run solves the temperature equation
  once per SIMPLE iteration and a coupled run once per outer iteration.
- Neither solver scales beyond about six cores on the reference machine.
- A measurement protocol and honest reproducibility bounds for every timing.



### Added

- `solveEnergy` (default `no`): a temperature transport equation
  `fvm::div(phi, T) - fvm::laplacian(alphaEff, T)` solved **segregated** after
  the coupled p-U solve of each outer iteration, on the conservative Rhie-Chow
  flux the coupled system produced rather than on a flux rebuilt from an
  interpolated velocity. `alphaEff = nu/Pr + nut/Prt`, with `nut` taken from
  the momentum-transport model immediately after its own correction. The
  equation is relaxed through `relaxationFactors { equations { T ...; } }` and
  solved through `solvers { T { ... } }`, so it uses exactly the machinery
  `k` and `epsilon` already use, and a `T` key in `residualControl` is picked
  up by the existing segregated-residual path with no new mechanism. See
  `doc/USAGE.md` 9.
- `Pr` (default `0.71`) and `Prt` (default `0.85`) in the `COUPLED`
  dictionary. They live there rather than in `constant/physicalProperties`
  because for a passive scalar they are coefficients of the transport
  equation rather than properties the viscosity model knows about - the same
  placement OpenFOAM's own `scalarTransport` function object uses for its
  `alphal` and `alphat`.
- `T` is constructed **only** when `solveEnergy` is on, so a case that does
  not ask for heat transfer needs no `0/T`, registers no extra field and takes
  the same code path as before. With the switch on and no readable `0/T` the
  run stops at start-up with an error that names the switch and lists the wall
  conditions available, rather than failing inside a field read.
- A `coupled energy check` log line, the temperature counterpart of the
  existing `coupled assembly check`: the integrated wall heat flux, the net
  enthalpy outflow, the net boundary diffusive influx and the imbalance
  between the last two as a percentage. Summing the assembled equation over
  every cell telescopes the internal faces away and leaves exactly those two
  boundary sums, so on a converged solution they must agree; the line is a
  permanent verification print, not debug output.
- `tutorials/heatedDuct`: a plane channel with isothermal walls at Re = 20 and
  Pr = 7, 8 000 cells, seconds to run, covered by the top-level `Allrun`. It
  is the worked example for `solveEnergy` and the physics gate for it.
- `tools/heated_duct_nusselt.py`: local Nusselt number of a plane channel from
  a written solution and the cell centres of
  `foamPostProcess -func writeCellCentres`.

### Changed

- The `COUPLED` sub-dictionary is now read **before** `createFields.H` rather
  than after, because `solveEnergy` decides whether `T` is constructed. No
  behavioural change; it is noted because it is the one ordering dependency
  `createFields.H` has on the main file.

### Findings

- **The physics gate passes.** Fully developed laminar flow between isothermal
  parallel plates has an exact Nusselt number of 7.5407, and `heatedDuct`
  reproduces **7.5416, +0.012 %**, monotonically from above through the
  thermal entry region and flat from x = 6 m. Repeating at
  `residualControl U 1e-10 / p 1e-10 / T 1e-12` gives the same four decimal
  places. The constant-heat-flux counterpart, Nu = 8.235, reproduces as 8.2466
  with the wall heat flux exactly equal to its analytical value of 2.5 K m3/s.
- **Against a segregated reference solving the identical discrete equation**
  - the stock `scalarTransport` function object with `alphal = 1/Pr` and
  `alphat = 1/Prt`, run on the coupled solver's own converged `phi` - the two
  temperature fields agree to max 4.05e-05 K and rms 2.78e-06 K on a field
  spanning 49.28 K, and the wall heat fluxes to 1.0e-08 relative. That
  residual is the coupled run's own `residualControl T 1e-6`, not a
  discretisation difference. Against a full segregated SIMPLE run the
  difference is 4.5e-02 K max, an order of magnitude above the reference's own
  convergence error of 3.7e-03 K, and it is the Rhie-Chow damping difference of
  `doc/VALIDATION.md` 3 arriving through `phi`: the inlet pressure drops differ
  by 0.17 %, in the middle of the documented laminar band.
- **The discrete energy balance closes at linear-solver tolerance.** The
  boundary convective outflow and diffusive influx agree to between 1.5e-11 %
  and 2.4e-09 % on every outer iteration, with no drift and with the sign of
  roundoff. With adiabatic walls the wall heat flux is exactly 0 and `T` stays
  at exactly its inlet value in every cell; with `fixedGradient` walls the flux
  is exactly the analytical `2*alpha*g*A_wall`.
- **Passive-scalar treatment is exact for this scope, not an approximation.**
  With constant properties and no buoyancy the momentum and continuity
  equations close on `U` and `p` alone, so solving `T` afterwards is the
  structure of the problem rather than a splitting error. Nothing in the
  energy step writes to `U`, `p` or `phi`, and this is verified rather than
  argued: with the switch absent and with `solveEnergy no` written explicitly,
  all four existing tutorials produce logs identical line for line to the
  pre-change build - 12, 25, 16 and 146 outer iterations, with
  `stripChannel` at 0.79284956 and `finnedDuct` at 0.5413873 to every printed
  digit.
- **A frozen-flux verification has to converge its deferred correction.** With
  a second-order upwind `div(phi,T)` the convection term is not linear in one
  pass even at fixed `phi`; five passes of the reference leave its residual at
  6.8e-05 and the comparison then reads 1.1e-02 K, which measures the
  reference and not the solver. Thirty passes take it to 9.5e-13.

## 0.10.0

### Added

- `adaptiveTolerance` (default `no`): an Eisenstat-Walker schedule for the
  PETSc inner relative tolerance, computed each outer iteration from the ratio
  of successive nonlinear block residuals and clamped into
  `[relTolMin, relTolMax]`, with `adaptiveGamma` and `adaptiveAlpha` setting
  the forcing term. Measured on `stripChannel` at `relTolMax 1e-3`: 2.9x fewer
  inner iterations and a 2.5x faster solve at np4, with the outer-iteration
  count, the pressure drop to every printed digit and machine-precision-class
  continuity all preserved. On the 2.15M-cell case of `doc/BENCHMARKS.md` 11 it
  also removed a convergence stall that a fixed inner `relTol` of 1e-6 never
  resolved, taking that run from 1112 s to 710 s. See `doc/USAGE.md` 4.7.
- `pcReuseInterval` (default `1`): periodic PETSc preconditioner refresh -
  rebuild on outer iterations `1, N+1, 2N+1, ...` and reuse in between, with
  the reuse flag explicitly cleared on rebuild iterations. Distinct from the
  existing `reusePreconditioner`, which freezes after the first build and
  still takes precedence when both are set.
- `coupledSolver::setRelTol`, an optional backend hook with an empty default
  implementation; the Eigen backends inherit it and ignore the schedule.
- A tuned configuration for large three-dimensional cases, written out in full
  in `doc/BENCHMARKS.md` 9.2: `adaptiveTolerance yes` with `relTolMax 3e-2`,
  `pcReuseInterval 8`, `-ksp_type fbcgs` in place of restarted FGMRES, the
  `a11` Schur approximation, and a BoomerAMG pressure block configured for
  three-dimensional problems. It is a documented recipe rather than a new
  default, and it is meant for three-dimensional cases of roughly a million
  cells and up; on small or two-dimensional cases the shipped defaults remain
  the better choice, where `relTolMax 3e-2` is too loose and neither
  preconditioner reuse nor the 3-D AMG settings pay for themselves on a
  hierarchy that is cheap to build.

### Changed

- The recommendation for large three-dimensional cases. Earlier releases
  reported the coupled solver as roughly 2.2 to 2.3 times slower than
  segregated on the 483k- and 2.15M-cell unit cells, and concluded that a
  crossover would need a preconditioner with a more nearly size-independent
  iteration count; those figures are now the starting point of the tuning
  recorded below rather than the standing verdict. Above roughly 300k cells the
  tuned configuration is the faster of the two, and above a million cells it
  wins by a factor of two or more. No shipped default moved with it: `selfp`,
  a fixed inner `relTol` and `pcReuseInterval 1` are still the defaults, and on
  small or two-dimensional cases they are still the right ones.

### Findings

- On the 2.15M-cell laminar interrupted-fin duct (8 601 600 unknowns, ten MPI
  ranks) the tuned configuration converges in 29 outer iterations and 203 s
  against the segregated solver's 555 iterations and 478 s, 2.3x faster, with
  dP 0.96255897 against 0.96269619 and an rms velocity difference over all
  2 150 400 cells of 1.26e-3, 0.075 % of the field maximum. That is the same
  difference the slow, tightly solved run of `doc/BENCHMARKS.md` 10 produced
  (1.28e-3), so none of the speed was bought with a looser answer, and
  normalised `|div(phi)|` stays at ~1e-9. Both timings repeat back to back:
  coupled 203 s / 203 s, segregated 480 s / 478 s. `doc/BENCHMARKS.md` 11.
- A four-point size sweep over the same geometry, both solvers at their best
  settings and every run formally converged, puts the crossover near 300k cells
  on a ten-core workstation: 269k cells 221 iterations / 13 s segregated
  against 21 / 14 s coupled, a tie; 533k 296 / 44 s against 24 / 35 s, 1.3x;
  1.07M 404 / 149 s against 25 / 75 s, 2.0x; 2.15M 555 / 478 s against
  29 / 203 s, 2.3x. The segregated count grows with the mesh while the coupled
  count is flat (21 -> 24 -> 25 -> 29), so the advantage compounds with size.
  Agreement improves with refinement rather than degrading - dP 0.18 % ->
  0.12 % -> 0.063 % -> 0.014 %, field rms 0.12 % -> 0.090 % -> 0.074 % ->
  0.075 % - which is what an O(h^2) Rhie-Chow damping difference should do.
  `doc/BENCHMARKS.md` 9.4.
- The factor of 5.5 between the first 2.15M-cell run and the tuned one is four
  compounding changes, each measured separately on that case: 1112 s baseline
  at a fixed `relTol` of 1e-6, which did not satisfy the convergence test at
  all, then 710 s with `adaptiveTolerance yes` and `relTolMax 1e-2`, 665 s with
  `pcReuseInterval 3`, 556 s with `-ksp_type fbcgs`, which removes
  orthogonalisation against a 30-vector Krylov basis of 8.6M unknowns, 354 s
  with the Schur-block AMG retuned for three dimensions, and 203 s with
  `pcReuseInterval 8`. `doc/BENCHMARKS.md` 9.1.
- The single largest lever was that AMG retuning. The Schur block had been
  running at PETSc's default `strong_threshold` of 0.25, which is a
  two-dimensional value; hypre's own guidance for three-dimensional problems is
  0.5, and the lower setting marks far too many connections strong, inflating
  operator complexity and the cost of every V-cycle. Correcting it, with HMIS
  coarsening and truncated `ext+i` interpolation, cut 556 s to 354 s, and the
  cheaper V-cycles are what made a long reuse interval pay; 8 to 12 is the
  optimum (206 s at 12, 238 s at 16).
- Preconditioner choice at scale. At 8.6M unknowns the shipped `selfp` Schur
  approximation costs 555 s per outer iteration, because it rebuilds an
  explicit sparse matrix product every outer iteration, while
  `-pc_fieldsplit_schur_precondition a11` costs 98.7 s by reusing the pressure
  block the assembly already builds, 5.6x cheaper; ASM is 112 s. `selfp` stays
  the default because it remains the more robust choice on small or highly
  non-orthogonal meshes. `doc/BENCHMARKS.md` 8.1.
- Monolithic block AMG in the Fluent style was measured and rejected; it is not
  outstanding work. Per outer iteration at 8.6M unknowns, with the residual
  after three outer iterations alongside: Schur `a11`, the adopted
  configuration, 98.7 s and 0.037; hypre nodal coarsening with block
  interpolation and a nodal Schwarz smoother 363-370 s and 0.058-0.061; hypre
  nodal systems-AMG 389 s and 0.977 with the Frobenius norm, 684 s and 0.968
  with the row sum; GAMG with unsmoothed aggregation and block Jacobi fails,
  diverging with NaN on the first application; the best block AMG with diagonal
  equation scaling 292 s and 0.114, while that same scaling hurt the adopted
  `a11` configuration, 98.7 s to 196.7 s per outer iteration. The architecture
  is directionally sound - block interpolation with a block smoother improved
  the three-iteration residual sixteenfold over plain nodal coarsening on the
  same matrix - but assembled from generic library components it costs three to
  four times more per outer iteration than the Schur splitting. A competitive
  version needs purpose-built coarsening, interpolation and smoothers, which is
  a solver project and not a configuration change. `doc/BENCHMARKS.md` 9.3.
- The gotcha that makes those measurements meaningful, recorded because it is
  silent: PETSc does not forward the matrix block size to hypre.
  `MatSetBlockSize(A, 4)` only sets the default value of an option whose hypre
  call is guarded by a flag test, so `-pc_hypre_boomeramg_numfunctions 4` has
  to be passed explicitly or hypre runs at `num_functions = 1` and every
  systems-AMG option degenerates without warning. Every run above passed it,
  and every run carried `-options_left`, which reported no unused options.

### Notes

- Both features are off by default and a run with neither key present - or
  with `adaptiveTolerance no` and `pcReuseInterval 1` written out - produces
  byte-identical logs to the previous release on the serial regression cases.
- The adaptive path switches the KSP convergence test to the initial-residual
  norm (`KSPConvergedDefaultSetUIRNorm`), which is what an Eisenstat-Walker
  forcing term is defined against. Under PETSc's default `||b||`-relative test
  a warm-started solve at a loose `rtol` returns without performing a single
  Krylov iteration. The fixed-`relTol` path is untouched.
- No tutorial result moved: `cavity` still converges in 12 outer iterations,
  `stripChannel` in 25 at a pressure drop of 0.79284956, `finnedDuct` in 16 and
  `stripChannelRAS` in 146.
- The findings above are laminar three-dimensional results. On RANS both
  solvers still floor out against the lagged turbulence, the coupled solver
  reaching a 4.7x deeper floor at comparable cost, so SIMPLE remains the
  production choice there, and there is still nothing to relax or tune in the
  coupled block itself.

## 0.9.0 - initial public release

### Added

- Fully coupled steady incompressible p-U solver for OpenFOAM 14 (Foundation):
  block-4 `[u, v, w, p]` system with an implicit pressure-gradient block and a
  Rhie-Chow continuity row, assembled as backend-agnostic COO triplets.
- Serial Eigen backend (`SparseLU`, `BiCGSTAB`) and MPI PETSc backend
  (`MATMPIAIJ` + `KSP`), selected automatically or through `linearSolver`.
- RANS support through OpenFOAM's `momentumTransport` framework, with the
  turbulence equations solved segregated and lagged one outer iteration.
- Pseudo-transient continuation, pressure-reference pinning, exact zero-flux
  treatment of impermeable constraint patches, and a `rhieChowFactor` switch
  for reproducing a relaxed segregated reference.
- Four tutorials (`cavity`, `stripChannel`, `finnedDuct`, `stripChannelRAS`),
  a field-comparison tool and an independent 2-D Python reference
  implementation of the same algorithm.
- Documentation: theory, installation, usage, benchmarks, validation,
  development notes and a bibliography.
