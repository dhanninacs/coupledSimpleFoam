# Validation

This document is the correctness evidence for `coupledSimpleFoam`: what has
been checked, by what method, what was measured, and - at least as important -
what has not been checked. Performance numbers live in `doc/BENCHMARKS.md`;
this file is only about whether the answers are right.

Sections 1 to 6 are the verification layers, section 7 is the defect history,
section 8 relates the observed behaviour to the published literature, section 9
is the explicit list of what is *not* covered, and section 10 explains how to
reproduce the whole set.

## Contents

1. [Verification strategy](#1-verification-strategy)
2. [Discrete mass conservation](#2-discrete-mass-conservation)
3. [Agreement with the segregated solver](#3-agreement-with-the-segregated-solver)
4. [Parallel consistency](#4-parallel-consistency)
5. [Cross-implementation check](#5-cross-implementation-check)
6. [Turbulence regression](#6-turbulence-regression)
7. [Defects found and fixed during verification](#7-defects-found-and-fixed-during-verification)
8. [Comparison with the published literature](#8-comparison-with-the-published-literature)
9. [What is not validated](#9-what-is-not-validated)
10. [Re-running the verification](#10-re-running-the-verification)

---

## 1. Verification strategy

A coupled assembly is easy to get almost right. It will converge, produce a
plausible flow field, and be wrong by a few per cent in an integral quantity.
No single check catches every way that can happen, so the solver is verified in
five layers, each of which fails on a different class of error.

**Layer 1 - an executable specification.** `tools/reference_prototype/`
contains an independent two-dimensional finite-volume implementation of the
same algorithm, in Python, written from the published treatments rather than
from the solver sources. It implements *both* a segregated SIMPLE loop and the
coupled block assembly over one discretisation, so the two differ only in
coupling strategy. It shares no code, no mesh representation and no linear
algebra with the C++ solver. It fixes the intended structure of the assembly -
the implicit Green-Gauss pressure-gradient block in the momentum rows, the
continuity row built from the Rhie-Chow flux and implicit in both velocity and
pressure, and the lagged explicit remainder on the right-hand side - before any
OpenFOAM code exists to be argued about.

**Layer 2 - algebraic operator equivalence.** Where a change replaces one
expression of an operator with another, the two are shown to be the same object
by unrolling OpenFOAM's own sources, rather than by comparing outputs and
declaring them close. This is what makes the turbulence-framework change
(section 6) verifiable at the bit level instead of at the plotting level.

**Layer 3 - discrete conservation.** Every outer iteration prints the summed
and maximum discrete divergence of the reconstructed face flux. This is the
most sensitive test of the continuity-row assembly: a wrong sign, a wrong
interpolation weight, or a dropped boundary term destroys it immediately and
visibly (section 2).

**Layer 4 - solver-to-solver comparison.** The converged field is compared
against OpenFOAM's own segregated solver on identical meshes, identical
schemes, identical boundary conditions and identical stopping criteria. This is
the layer that catches errors which conserve mass perfectly while solving the
wrong momentum equation (section 3).

**Layer 5 - parallel-versus-serial consistency.** The same case is run
serially through a direct sparse factorisation and in parallel through PETSc,
and the fields are compared cell by cell. This isolates the processor-patch
coupling and the remote index exchange, which are the only pieces of code that
exist solely for the parallel path (section 4).

The layers are complementary, and the ordering matters: layer 3 verifies that
the flux reconstruction and the assembled continuity row agree with each other,
but not that either is correct; layer 4 verifies the answer but is limited by
how well the reference itself is converged; layer 5 verifies the parallel code
against the serial code but not the physics. The one defect that survived
several of these layers is described in section 7.1, and it survived exactly
because it was consistent with itself.

Every check below is either a case under `tutorials/` or a script under
`tools/`, so none of it is a claim that has to be taken on trust.

---

## 2. Discrete mass conservation

### The check

After every coupled solve the solver rebuilds the face flux `phi`, sums it over
the faces of each cell, and prints:

```
time step continuity errors : sum local = 1.28080978574e-16, global = -8.98498470606e-19, cumulative = -8.98498470606e-19
coupled assembly check: sum|div(phi)| = 1.28080978574e-16 (excl. pRef cell 1.2643671557e-16), max|div(phi)| = 1.64426300419e-18, normalised = 1.75162884398e-16
```

The `normalised` figure divides the summed absolute divergence by the summed
absolute face flux over the whole mesh, so it is dimensionless and comparable
across cases. All three sums are reduced across ranks, so the parallel figure
is a global one.

### Measured values

Lid-driven cavity, 64 x 64, Re = 100, first outer iteration:

| quantity | value |
| --- | --- |
| `sum abs(div(phi))`, all cells | 1.28e-16 |
| `sum abs(div(phi))`, excluding the pinned cell | 1.26e-16 |
| `max abs(div(phi))` | 1.64e-18 |
| normalised by the characteristic flux | 1.75e-16 |

The acceptance requirement was `< 1e-10` times the characteristic flux; the
measurement is six orders of magnitude inside it, and stays at that level for
every subsequent outer iteration rather than only at the start.

The same holds with a turbulent, strongly varying effective viscosity. On the
backward-facing step with `kEpsilon`, `sum abs(div(phi))` is 2.2e-17 at
iteration 1 and 1.4e-16 at iteration 779 - unchanged in order of magnitude over
the whole run.

Note that the full sum and the sum excluding the pinned reference cell agree to
roundoff. When the pressure level is pinned, the continuity row of the
reference cell is replaced by an identity row and is therefore not enforced;
but the other N-1 rows are satisfied exactly and the net boundary flux of a
closed domain is zero, so the pinned cell's divergence has nowhere to go and
comes out at zero as well. A meaningful gap between the two numbers would
indicate that the boundary fluxes do not balance.

### Why it holds by construction

The continuity row of the coupled system is the sum over the cell's faces of

```
phi_f = ( w*U_P + (1-w)*U_N ) . Sf              [implicit velocity]
      - Df_f * |Sf| * Delta_f * ( p_N - p_P )   [implicit pressure]
      + Df_f * ( grad(p)_f . Seff_f )           [lagged, on the RHS]
```

After the solve, `phi` is rebuilt from *exactly* this expression, using the same
`Df`, the same interpolation weights, the same `nonOrthDeltaCoeffs` and the same
*lagged* pressure gradient that entered the matrix. The pressure gradient is
only refreshed afterwards, for the next iteration. The divergence of the
rebuilt flux is therefore the residual of the continuity rows, which the linear
solver has just driven to its tolerance.

That makes this the single most sensitive test of the continuity assembly. Any
discrepancy between the assembled row and the reconstruction expression - a
flipped sign on a boundary term, `0.5` used where `mesh.weights()` belongs,
`deltaCoeffs` used where `nonOrthDeltaCoeffs` belongs, a patch loop that skips a
patch in one place but not the other - shows up as a divergence many orders of
magnitude above machine precision, on the first iteration, with no ambiguity
about whether it is "close enough".

It is also live rather than tautological, because it detects an inner solve that
did not actually converge. With `relTol 1e-6` on the RANS backward-facing step,
`sum abs(div(phi))` degrades to 1.8e-09 (normalised 2.8e-08) instead of machine
precision, and the outer loop fails to finish within 1000 iterations; at the
shipped `relTol 1e-12` continuity returns to ~3e-15. The check is a direct
readout of how well the coupled system is being solved.

### What it does not test

Conservation by construction cuts both ways. The check verifies that the
reconstruction and the assembled row agree with each other and that the linear
solve converged. It does **not** verify:

- **the momentum rows.** An error in the momentum block - a double-counted
  patch diagonal, a wrong pressure-gradient weight - leaves the continuity rows
  and the flux reconstruction untouched, so the divergence stays at 1e-16 while
  the velocity field is wrong. Section 3 is the layer that catches this.
- **whether the shared expression is the right one.** If a boundary contributes
  a flux it physically should not, and the same wrong flux appears in both the
  matrix row and the reconstruction, the divergence is still exactly zero. This
  is not hypothetical: it is precisely how the constraint-patch mass leak of
  section 7.1 stayed invisible to this check while corrupting an integral
  quantity by 9.6 per cent.

---

## 3. Agreement with the segregated solver

The reference is `foamRun -solver incompressibleFluid` (SIMPLE, relaxation
U 0.7 / p 0.3) on the same mesh, the same `fvSchemes`, the same `0/` and
`constant/` directories, and the same `residualControl`. The coupled runs use no
relaxation factors at all; they simply read the additional `COUPLED`
sub-dictionary in `system/fvSolution`, which `foamRun` ignores.

### The reference must itself be deeply converged

This is the methodological point that governs every number in this section, and
getting it wrong makes a correct solver look inaccurate.

Near convergence SIMPLE's residuals on the 64 x 64 cavity decay by about 0.993
per iteration. Stopping at `U 1e-5` therefore still leaves on the order of 1e-3
of *solution* error in the reference field. Comparing against such a reference
does not measure the difference between the two solvers; it measures how far the
reference still is from its own answer. Both levels are reported below so the
effect is visible rather than assumed.

| comparison | centreline max abs(dUx) | all-cell max abs(dUx) |
| --- | --- | --- |
| coupled vs segregated at U 1e-5 / p 1e-4 (755 SIMPLE iterations) | 8.82e-4 | 1.40e-2 (lid corner cell) |
| coupled vs segregated at U 1e-9 / p 1e-9 (2,021 SIMPLE iterations) | **1.03e-5** | 1.40e-2 (lid corner cell) |
| coupled with `rhieChowFactor 0.7` vs the tight reference | **1.43e-6** | 9.60e-4 |

Tightening the reference alone improves the apparent agreement by a factor of
86, without changing anything in the coupled solver. The acceptance requirement
was `< 1e-3`; the measurement against a properly converged reference is 1.03e-5.

The same effect at the next mesh level: tightening the 128 x 128 cavity
reference to roughly 6,000 iterations collapses the centreline difference to
2.1e-5.

The segregated iteration counts are not themselves a stable quantity. The same
64 x 64 cavity at nominally identical `residualControl` has been measured at 755
and at 797 SIMPLE iterations in separately built versions of the case; small
differences in dictionary details move the count. Neither number should be
treated as canonical, which is one more reason to judge agreement against a
deeply converged reference rather than against an iteration count.

### The measured profile

Vertical-centreline `Ux` against the tight (1e-9) reference, taken as the mean
of the two cell columns adjacent to x = 0.5:

```
           y               Ux_coupled          Ux_segregated       diff
   0.0078125  -6.0026875307e-03  -6.0030217757e-03      3.342e-07
   0.0703125  -4.6732653038e-02  -4.6730684320e-02     -1.969e-06
   0.1328125  -8.0309838634e-02  -8.0307071852e-02     -2.767e-06
   0.1953125  -1.1065942688e-01  -1.1065574919e-01     -3.678e-06
   0.2578125  -1.3945844746e-01  -1.3945379972e-01     -4.648e-06
   0.3203125  -1.6612209214e-01  -1.6611667791e-01     -5.414e-06
   0.3828125  -1.8775154520e-01  -1.8774593508e-01     -5.610e-06
   0.4453125  -1.9947481180e-01  -1.9946978192e-01     -5.030e-06
   0.5078125  -1.9559665253e-01  -1.9559286096e-01     -3.792e-06
   0.5703125  -1.7147718867e-01  -1.7147505560e-01     -2.133e-06
   0.6328125  -1.2539557036e-01  -1.2539511245e-01     -4.579e-07
   0.6953125  -5.8731598769e-02  -5.8732593489e-02      9.947e-07
   0.7578125   2.7751956716e-02   2.7749458598e-02      2.498e-06
   0.8203125   1.4388253542e-01   1.4387787907e-01      4.656e-06
   0.8828125   3.2492519267e-01   3.2491733763e-01      7.855e-06
   0.9453125   6.3073726746e-01   6.3072742670e-01      9.841e-06
```

### Integral quantities across the laminar suite

Pressure drop (`dp`) at production `residualControl`, coupled against
segregated on identical meshes:

| case | cells | segregated -> coupled | difference |
| --- | --- | --- | --- |
| backward-facing step, laminar | 12,225 | 43.90 -> 43.74 | 0.37 % |
| strip channel, Re 400 | 10,920 | 1.3209 -> 1.3186 | 0.17 % |
| strip channel, Re 1000 | 10,920 | 0.7959 -> 0.7928 | 0.38 % |
| 3-D interrupted-fin unit cell | 483,000 | 670.44 -> 669.68 | 0.11 % |

The 3-D figure is quoted with `rhieChowFactor 0.85`; at the default 1.0 it is
1.5 per cent, for the reason given next.

### The Rhie-Chow damping difference, and the experiment that isolates it

The residual difference between the two solvers is not noise and is not an
error. It has a single identified cause, and there is a one-key experiment that
collapses it.

Segregated SIMPLE forms `rAU = 1/UEqn.A()` from the **relaxed** momentum matrix,
so its converged Rhie-Chow damping is effectively `alpha_U * V/aP`, not `V/aP`.
A coupled solver has no equation relaxation factor, so `D_f = V/aP` is the
natural choice - and it is what the reference prototype does. The two
discretisations therefore differ by a Rhie-Chow term of order `h^2`, which is
invisible in smooth regions and becomes visible exactly where `rAU` jumps
hardest between neighbouring cells.

On the cavity that place is the singular lid corner, which is why the all-cell
difference (1.40e-2) is two orders of magnitude larger than the centreline
difference (1.03e-5) while the reference is the same.

Setting `rhieChowFactor` to the reference run's momentum relaxation factor
reproduces the segregated damping. The effect has been quantified in three
independent settings:

| case | `rhieChowFactor` 1.0 | matched to the reference | reduction |
| --- | --- | --- | --- |
| cavity 64 x 64, all-cell max abs(dUx) | 1.40e-2 | 9.60e-4 at 0.7 | 15x |
| cavity 64 x 64, centreline max abs(dUx) | 1.03e-5 | 1.43e-6 at 0.7 | 7x |
| backward-facing step, RANS, `dp` gap | 0.678 % | 0.075 % at 0.7 | 9x |
| 3-D interrupted-fin unit cell, `dp` gap | 1.5 % | 0.11 % at 0.85 | 14x |

The RANS row is the largest effect, and for a comprehensible reason: `nuEff`
spans orders of magnitude in a RANS field, so `rAU` jumps much harder between
neighbours than in a laminar one. That row also costs essentially nothing in
convergence - 762 outer iterations at `rhieChowFactor 0.7` against 779 at 1.0.

Two caveats keep this honest. First, the RANS gap is measured against a
segregated reference that never actually meets its own tight criterion (SIMPLE's
`p` residual floors at 1.8e-04 on that case) and whose inlet pressure wanders
over a band of about 0.005 per cent; the remaining 0.075 per cent is above that
band, so it is real, but the reference has a noise floor. Second, the *loose*
segregated reference on the same case sits 0.06 per cent from its own converged
answer - the same size as the entire remaining coupled-versus-segregated gap.
Judging `dp` against a loosely converged SIMPLE run is meaningless at this
level.

The residual difference that survives matching `rhieChowFactor` - 9.6e-4 at the
cavity corner cell, 0.075 per cent on the RANS step - is a second, smaller
formulation difference. OpenFOAM's `phiHbyA` uses `fvc::flux(rAU*grad(p))`, with
`rAU` inside the interpolation, whereas the coupled assembly uses
`Df_f * (interpolate(grad p) . Seff_f)`. These agree to order `h^2` and differ
most where `rAU` jumps between neighbours - the same locations, for the same
reason.

The default remains `rhieChowFactor 1.0`, because it is the correct choice for a
solver with no relaxation factor. The key exists to reproduce a segregated
reference for verification, not for production use.

---

## 4. Parallel consistency

The assembly is identical in serial and in parallel. It emits backend-agnostic
COO triplets carrying *global* row and column indices, which the Eigen backend
consumes serially and the PETSc backend consumes across ranks. The only code
that exists solely for the parallel path is the processor-patch contribution
(`+internalCoeffs` on the diagonal, `-boundaryCoeffs` into the remote column)
and the exchange of remote global cell indices and remote field values.

### Outer-iteration counts are rank-independent

| case | cells | unknowns | serial Eigen | np = 4 | np = 6 |
| --- | --- | --- | --- | --- | --- |
| cavity 64 x 64 | 4,096 | 16,384 | 12 | 12 | - |
| cavity 128 x 128 | 16,384 | 65,536 | 12 | 12 | 12 |
| cavity 256 x 256 | 65,536 | 262,144 | 12 | 12 | 12 |
| strip channel, Re 1000 | 10,920 | 43,680 | 25 | 25 | 25 |
| backward-facing step, laminar | 12,225 | - | 22 | 22 | - |

Identical, not plus or minus one. Decomposition is `scotch`.

### Field agreement

| comparison | centreline max abs(dUx) | all-cell max abs(dU) |
| --- | --- | --- |
| cavity 64 x 64, np4 vs serial Eigen SparseLU | 9.00e-12 | 1.21e-11 |
| cavity 128 x 128, np4 vs serial Eigen SparseLU | 3.70e-11 | 5.71e-11 |
| cavity 128 x 128, np6 vs serial Eigen SparseLU | 3.40e-11 | 5.01e-11 |
| cavity 256 x 256, np4 vs serial Eigen SparseLU | 2.05e-10 | 2.09e-10 |
| cavity 256 x 256, np6 vs serial Eigen SparseLU | 2.03e-10 | 2.07e-10 |
| cavity 64 x 64, serial PETSc vs serial Eigen | 9.00e-12 | 9.55e-12 |

The acceptance requirement was 1e-8; the measurement is 1e-10 or better. Part of
the residual floor is the ASCII `writePrecision 12` in `controlDict`, i.e. about
1e-12 on values of order one, so these figures are close to what the file format
can express.

On the strip channel the parallel pressure drop reproduces the serial value to
**all printed digits** (7.92849560e-01) at both 4 and 6 ranks.

### What this establishes

Two distinct things, which the last row of the table separates:

- **The backend is not the physics.** Serial PETSc against serial Eigen
  SparseLU - an iterative Krylov solve against a direct sparse factorisation, on
  the same assembled matrix - agrees to 9e-12. Whatever the parallel runs show
  is therefore about decomposition, not about the linear solver.
- **The processor-patch coupling is correct.** The sign convention on coupled
  patches follows `processorFvPatchField::updateInterfaceMatrix`
  (`result[faceCells] -= coeffs*pnf`), which is why `boundaryCoeffs` enters the
  remote column negated while `internalCoeffs` enters the diagonal positively.
  Getting that sign wrong, or mismatching the remote global indices returned by
  `syncTools::swapBoundaryCellList`, does not produce a small discrepancy - it
  produces a qualitatively different solution and a different iteration count.
  Reproducing the serial answer to 1e-10 with a bit-identical iteration count,
  across three mesh sizes and two rank counts, is the empirical proof that both
  are right.

The rank-independence of the iteration count is also a statement about the
outer loop rather than the inner one: the Picard contraction rate does not
depend on how the mesh was cut.

### RANS is close but not bit-identical

On the RANS backward-facing step, serial and np4 give 779 and 771 outer
iterations (about 1 per cent apart) with inlet pressures matching to 0.006 per
cent. The laminar cases match exactly and the RANS case does not, because at
these tolerances PETSc's non-reproducible floating-point reductions move the
iteration at which the criteria are met by a few. The physics is the same; the
stopping point differs by rounding.

---

## 5. Cross-implementation check

`tools/reference_prototype/coupled_vs_simple.py` is a two-dimensional
finite-volume implementation of the same algorithm, in Python with NumPy and
SciPy, sharing nothing with the C++ solver: different language, different mesh
representation, different linear algebra, no OpenFOAM involvement at all. It
solves a collocated 2-D channel with three interrupted fins blocked out of a
structured mesh, twice on the same mesh with the same discretisation
(first-order upwind convection, central diffusion, Rhie-Chow face flux) - once
with a segregated SIMPLE loop and once with the coupled block assembly.

### Iteration behaviour

| grid, Re | SIMPLE (0.7 / 0.3) | coupled (no relaxation) | ratio |
| --- | --- | --- | --- |
| 120 x 24, Re 400 | 149 | 29 | 5.1x |
| 240 x 48, Re 400 | 284 | 27 | 10.5x |
| 240 x 48, Re 1000 | 241 | 31 | 7.8x |
| any grid, relaxation 0.85 / 0.5 | diverged | not applicable | - |

### Pressure drop against the C++ solver

Same geometry, Re 400, both codes running both algorithms:

| | segregated | coupled |
| --- | --- | --- |
| Python reference implementation | 1.3215 | 1.3198 |
| `coupledSimpleFoam` / `foamRun` | 1.3209 | 1.3186 |

The four values span about 0.2 per cent. Two separately written codebases, each
running both a segregated and a coupled algorithm, land on the same answer.

### What agreement across implementations establishes

- **The C++ assembly realises the intended algorithm.** The two codes share no
  source. An indexing error, a transposed contribution, or a misplaced weight in
  the C++ block assembly would have to be reproduced independently in Python to
  escape detection.
- **The convergence signature is a property of the algorithm, not of
  OpenFOAM.** Both codes show the same qualitative behaviour: segregated
  iteration counts grow under mesh refinement (149 -> 284) while coupled counts
  do not (27 to 31); and SIMPLE diverges at relaxation 0.85 / 0.5 where the
  coupled solver needs no relaxation at all. The C++ solver reproduces the same
  signature on OpenFOAM meshes - 12 outer iterations at 64 x 64, 128 x 128 and
  256 x 256 cavities, a 16-fold range of cell count.
- **The absolute answer is not an artefact of either code.** Four values from
  two codebases and two algorithms agreeing to 0.2 per cent is a much stronger
  statement than one code converging without crashing.

### The limits of this check

The prototype is an *executable specification*: it was written first, and the
C++ assembly follows it. Agreement therefore shows that the C++ code correctly
implements the specified algorithm - it does not protect against an error in the
specification itself, which is what layers 3 and 4 are for. The prototype is
also structured, two-dimensional and orthogonal, so it exercises none of the
unstructured, non-orthogonal, boundary-condition-polymorphic machinery that
dominates the C++ assembly.

---

## 6. Turbulence regression

### The requirement

Turbulence support replaced a hand-rolled constant-viscosity treatment with
OpenFOAM's own `momentumTransport` framework, so that the momentum operator
became literally the one the stock `incompressibleFluid` solver module builds:

```cpp
fvVectorMatrix UEqn
(
    fvm::div(phi, U)
  + momentumTransport->divDevSigma(U)
);
```

The gate for that change is that it must leave laminar results **unchanged** -
not close, unchanged. A laminar case run before and after must produce the same
matrix, and therefore the same iteration path and the same log.

### The operator-equivalence argument

Rather than infer equivalence from similar output, the two operators are shown
to be the same object by unrolling the OpenFOAM 14 sources:

```
divDevSigma(U) = divDevTau(U) = DivDevTau(rho_=1, U)
               = divDevTauCorr(-alphaRhoNuEff*fvc::dotInterpolate(Sf, dev2(T(grad U))), U)
                 - fvm::laplacian(alphaRhoNuEff, U)
divDevTauCorr(X,U) = fvm::divc(surfaceVectorField("devTauCorrFlux", X), U)
fvm::divc(tflux, vf)  = fvm::Su(fvc::div(tflux), vf)
fvm::Su(su, vf)       : source() -= mesh.V()*su, and nothing else
alphaRhoNuEff         = fvc::interpolate(alpha_*rho*nuEff())
                      = fvc::interpolate(nu)   for a laminar Stokes model with
                                               a constant viscosityModel
```

Three identities close the argument:

1. `fvc::interpolate(uniform nu) == nuf` **bitwise**. OpenFOAM's weighted
   interpolation is coded as `lambda*(v_P - v_N) + v_N`, which returns `v`
   exactly for a uniform field, so no `w + (1-w) != 1` rounding creeps in.
2. `fvc::dotInterpolate(Sf, T) == Sf & fvc::interpolate(T)` - the same weights
   and the same `interpolationSchemes` lookup, fused into one loop.
3. `fvm::Su(fvc::div(flux), U)` writes `-V*fvc::div(flux)` into `source()` and
   contributes nothing to `diag`, `upper`, `lower`, `internalCoeffs` or
   `boundaryCoeffs`.

So for a constant-viscosity laminar case the new operator *is* the old
"divergence minus Laplacian plus an explicit `dev2(T(grad U))` flux". For a RAS
model the identical expression carries `nuEff = nu + nut` instead of `nu`, plus
the wall functions on `nut`. Everything downstream of the operator - coefficient
extraction, `aP`, `rAU`, `Df`, the pseudo-transient diagonal, `rhieChowFactor`,
the Rhie-Chow assembly, the flux reconstruction, the zero-flux patch rule and
the residual machinery - is operator-agnostic and was not modified.

### The measured outcome

Not "close": **bit-for-bit identical logs**, on two different topologies.

| case | backend / ranks | outer iterations | check |
| --- | --- | --- | --- |
| lid-driven cavity 64 x 64 (4,096 cells) | Eigen SparseLU, serial | 12 (was 12) | every `Solving for`, `continuity errors` and `assembly check` line identical to the pre-change binary; `diff` empty |
| strip channel, Re 1000 (inlet / outlet / wall / fin patches) | Eigen SparseLU, serial | 25 (was 25) | identical to the pre-change binary's log; `diff` empty |
| strip channel, Re 1000 | PETSc, np4 | 25 (was 25) | `areaAverage(inlet) of p = 0.79284956`, matching the serial direct-solve reference `7.92849560e-01` to all printed digits |

The **serial strip-channel** check is the decisive one. Eigen SparseLU is a
deterministic direct solve, so a bit-identical log over 25 outer iterations on a
case carrying inlet, outlet, wall and fin patches proves the assembled matrix is
unchanged - every diagonal, every off-diagonal, every boundary coefficient. A
cavity alone would not prove it, because it has one boundary-condition type.

The np4 logs agree only to about 7 digits, because PETSc fieldsplit with
BoomerAMG is not bitwise reproducible from run to run; the *initial* residuals
still match to 8 digits and the answer to all printed digits.

Two further gates were met when turbulence is actually active, on the
backward-facing step with `kEpsilon`: `k` and `epsilon` solve every outer
iteration with monotonically decreasing initial residuals (1 down to 9.8e-06
over 779 iterations), with zero `bounding k` or `bounding epsilon` warnings, no
NaN and no floating-point exception; and `sum abs(div(phi))` stays at ~1e-16
throughout, so the coupled continuity rows are still satisfied to machine
precision under a strongly varying `nuEff`.

---

## 7. Defects found and fixed during verification

Two assembly defects and one configuration defect were found and corrected
during verification. They are recorded here because each one is a trap a
reimplementation would fall into, and because the way each was detected says
something about which check is load-bearing.

### 7.1 Constraint-patch mass leak on non-axis-aligned faces

**Symptom.** The first run of the 483,000-cell 3-D interrupted-fin unit cell
(snappyHexMesh, maximum non-orthogonality 80.9 degrees, 5,408 skew faces)
converged cleanly in 36 outer iterations with a healthy residual history - and
gave a pressure drop 9.6 per cent below the segregated reference. Nothing in the
run looked wrong.

**Root cause.** `symmetry` patches do not report `fixesValue()`. The continuity
row therefore treated them like any other uncoupled patch and gave them velocity
and Rhie-Chow pressure contributions built from per-component transform
coefficients. Those coefficients are diagonal per component, and a diagonal
per-component representation cannot express "exactly zero flux through this
plane" when the face normal is not aligned with a coordinate axis. On an
axis-aligned mesh the representation happens to be exact; on tilted
snappyHexMesh faces it leaks mass through the symmetry plane.

**Why the 2-D suite did not catch it.** Every 2-D validation mesh is
axis-aligned, so the error is identically zero there. The defect is only
reachable on an unstructured 3-D mesh with tilted constraint-patch faces.

**Why the conservation check did not catch it.** This is the important part. The
leaked flux appeared in the assembled continuity row *and* in the flux
reconstruction, because both are built from the same expression. The two were
consistent with each other, so `sum abs(div(phi))` stayed at machine precision
throughout - the solver was perfectly conserving a boundary condition that was
wrong. What caught it was layer 4: comparison of an integral quantity against
the segregated solver.

**Fix.** A `zeroFluxPatch()` predicate identifies impermeable constraint patches

```cpp
zeroFluxPatch(patchi) =  isA<symmetryFvPatch>(patch)
                      || isA<symmetryPlaneFvPatch>(patch)
                      || isA<wedgeFvPatch>(patch)
                      || isA<slipFvPatchVectorField>(U.boundaryField()[patchi])
```

and forces both the continuity contribution and the reconstructed flux to
exactly zero on those patches, treating them like walls.

**Measured effect.**

| case | before | after |
| --- | --- | --- |
| 3-D interrupted-fin unit cell, `dp` gap | 9.6 % | 1.5 %, and 0.11 % once `rhieChowFactor` is matched |
| 3-D duct with a half-span fin, `dp` gap | 0.86 % | 0.54 % |
| cavity 128 x 128 regression | 12 iterations | 12 iterations, unchanged |

After the fix the entire remaining discrepancy on the 3-D case is attributable
to the documented Rhie-Chow damping definition of section 3, which is the
outcome that says the fix is complete rather than merely improving.

### 7.2 Patch diagonal counted twice

**The hazard.** Two different diagonal quantities are in play and they are easy
to conflate:

- `UEqn.diag()` is the raw `lduMatrix` diagonal. It does *not* include boundary
  contributions.
- `aP = UEqn.diag() + sum over patches of UEqn.internalCoeffs()`, accumulated
  onto the patch face-cells - exactly what `fvMatrix::A()` computes - plus the
  pseudo-transient diagonal when that is enabled. This is used **only** to form
  `Df = rhieChowFactor * V / cmptAv(aP)` for the Rhie-Chow face flux.

The coupled momentum diagonal is assembled in two places: the cell loop adds
`UEqn.diag()`, and the patch loop adds `UEqn.internalCoeffs()` for every patch,
including coupled ones. Seeding the cell loop from `aP` instead - which looks
like a harmless simplification, since `aP` is already computed and is "the
diagonal" in the sense `fvMatrix::A()` means - double-counts the boundary
contribution on every patch face cell.

**Symptom.** The run still converges. Mass conservation is unaffected, because
the momentum rows do not enter the divergence check at all (section 2). The
velocity field is wrong in the cells adjacent to boundaries, in proportion to
the boundary coefficient, which corrupts wall shear, inlet development and every
integral quantity derived from them - while every diagnostic the solver prints
looks healthy.

**Detection.** Layer 4 only. Comparison against the segregated solver on a case
with a high boundary-cell fraction is what exposes it; a coarse cavity, where
most of the flow is boundary-adjacent, is more sensitive to it than a large
interior-dominated mesh.

**Invariant now enforced.** The cell loop takes only the raw `lduMatrix`
diagonal; `internalCoeffs` enters exactly once, in the patch loop; `aP` is used
for `Df` and for nothing else. The assembly carries an explicit note to that
effect at the cell loop, because the two quantities have deceptively similar
names and a well-intentioned refactor reintroduces the defect.

### 7.3 A linear-solver default validated on one mesh topology only

Not an assembly defect, but a verification finding worth the same weight,
because it changed a shipped default.

The initial default PETSc configuration - FGMRES with `PCFIELDSPLIT` Schur
(`selfp`) and GAMG on the Schur block - was chosen on cavity-class meshes, where
it is about 1.9 times faster than the alternatives. On the strip-channel
topology it **diverges**, serially and in parallel: the inner KSP hits
`DIVERGED_ITS` at 5,000 iterations and the outer loop never converges. The
mirror image also holds: ASM (overlap 2) with sub-ILU(2) is robust on every 2-D
topology tested and caps out on the 483,000-cell 3-D mesh at 5,000 inner
iterations and roughly 440 seconds per outer iteration.

Each configuration is robust exactly where the other fails. The resolution was
hypre BoomerAMG on the Schur block, which is the only configuration measured to
converge on every topology tested - reproducing the serial direct-solve
reference to all printed digits on the strip channel where GAMG-on-Schur
diverges, and running the 3-D case at 3.7 to 3.8 seconds per outer iteration
against 12.2 with GAMG. That is the shipped default.

Two lessons carry forward:

- **A preconditioner default tuned on one mesh topology is not validated.** It
  is validated on that topology. The redeeming feature here is that the failure
  is loud - `DIVERGED_ITS` at the iteration cap, not a quietly wrong answer.
- **`PETSC_OPTIONS` alone cannot be used to test a preconditioner.** The
  `petscOptions` dictionary entry has a non-empty default, and it is inserted
  with `PetscOptionsInsertString` *after* `PetscInitialize` has already consumed
  the environment variable. The dictionary value therefore wins. An experiment
  that only exports `PETSC_OPTIONS` and changes nothing in `fvSolution` is a
  null experiment that silently measures the default. Set `petscOptions` in the
  `COUPLED` dictionary instead.

---

## 8. Comparison with the published literature

No claim here is a formal benchmark against digitised reference data; these are
qualitative agreements with published behaviour, which is what they are worth.

**Lid-driven cavity, Ghia et al.** At Re = 100 the segregated reference matches
the standard cavity benchmark for orientation (u_min about -0.199 at y about
0.45, u about 0.63 at y about 0.945), and the coupled solution matches that
segregated reference to 1.03e-5 on the centreline. The coupled solver therefore
agrees with the published benchmark to whatever degree OpenFOAM's own segregated
solver does, which is the meaningful statement given that both use the same
discretisation.

**Coupled-solver behaviour: Darwish, Sraj and Moukalled (JCP 228, 2009), and
Moukalled, Mangani and Darwish (2016), chapter 15.** The published coupled
assembly predicts three things, all three of which are reproduced:

1. *A large reduction in outer iterations.* Measured at 7.8x to 200x across the
   laminar suite (for example 12 against 2,400 on the 128 x 128 cavity, 22
   against 843 on the backward-facing step, 20 against 142 on the 3-D unit
   cell).
2. *Outer-iteration counts that are largely mesh-independent.* Measured at 12
   outer iterations on cavities of 4,096, 16,384 and 65,536 cells - a 16-fold
   range - while SIMPLE goes from 797 to 2,400 over the first factor of four
   alone. The reference prototype shows the same signature independently
   (149 -> 284 for SIMPLE under refinement, 27 to 31 for coupled).
3. *No under-relaxation factors.* The solver has no p-U relaxation factor at all
   and converges on every geometry tested without one, including cases where the
   segregated algorithm diverges at moderately aggressive relaxation.

**Block-coupled AMG: Uroic and Jasak (2018, 2021).** The published position that
a saddle-point block system needs a block-aware multigrid, rather than a scalar
one, is confirmed negatively here: plain GAMG applied to the whole stabilised
block system failed outright (killed at 20 minutes for a single outer
iteration), and hypre's scalar BoomerAMG on the whole system diverges at the
inner iteration cap. What works is a scalar AMG on the *Schur complement*, which
is the route this solver takes.

**Pseudo-transient continuation: ANSYS Fluent Theory Guide section 23.4.3.2 and
User's Guide section 37.3.** The documented role of pseudo-transient
continuation as a robustness device in place of under-relaxation is reproduced:
pseudo-transient continuation combined with a loose inner tolerance is 1.9 times
faster than a tight inner tolerance at the same outer-iteration count, and the
control run without damping at the same loose inner tolerance plateaus instead
of converging. Damping is what converts loose inner solves into convergence.

**Prior art: foam-extend `pUCoupledFoam`.** A freely available block-coupled
pressure-velocity solver exists in foam-extend (Jareteg, Vukcevic and Jasak,
2014). It is built on foam-extend's `BlockLduMatrix` infrastructure and cannot
be used with Foundation OpenFOAM 14, so it is prior art rather than a
comparison baseline reproducible from this repository.

---

## 9. What is not validated

This section is deliberately exhaustive. Anything not listed in sections 2 to 6
should be assumed untested.

### Physics and equation scope

- **Steady only.** There is no `fvm::ddt` term; cases must use
  `ddtSchemes { default steadyState; }`. Pseudo-transient continuation is a
  relaxation device, not a time-accurate scheme, and nothing about it has been
  validated as time accuracy.
- **Incompressible, single-phase, constant density.** The pressure is
  kinematic. No compressible support, no multiphase, no free surface.
- **No energy equation**, no conjugate heat transfer, no buoyancy.
- **No formal order-of-accuracy or grid-convergence study.** The discretisation
  is OpenFOAM's own, and the verification strategy is equivalence to the
  segregated solver rather than an independent demonstration of formal order.
  No Richardson extrapolation or mesh-refinement error study has been performed.
- **No comparison against experimental data.** Every comparison here is
  code-to-code.

### Turbulence

- **Only `kEpsilon` and `kOmegaSST` have been exercised.** `laminar` / `Stokes`
  is covered by the bit-for-bit regression gate of section 6.
- **"Any incompressible RAS or LES model is selectable" is an inference, not a
  tested claim.** It follows from the run-time selection tables of the linked
  libraries being complete, and the libraries linked
  (`libmomentumTransportModels`, `libincompressibleMomentumTransportModels`,
  `libphysicalProperties`) do carry every model and wall function. No model
  outside the two named above has been run.
- **The `kOmegaSST` evidence is thin.** It amounts to one 2-D case at
  laminar-transitional conditions, where the model correctly relaminarises to
  the laminar answers to three digits, plus one demonstration in a genuinely
  turbulent regime. Resolved-wall SST at transitional Reynolds numbers on 3-D
  unstructured meshes is pathological for the segregated solver and for this one
  alike - that is a statement about the flow regime, not about either solver,
  but it means SST is unvalidated there.
- **Turbulence is segregated and lagged** one outer iteration behind the coupled
  p-U solve. This sets the asymptotic convergence rate and is not a shortcut
  past the physical relaxation of the turbulence field. Pulling k and epsilon
  into the block system is not implemented.
- **`predict()` is not called** on the turbulence model. This is correct for
  every eddy-viscosity model in OpenFOAM 14, whose only non-trivial `predict()`
  overrides are no-ops for eddy-viscosity RAS, but a model whose `predict()` did
  real work would not get it.

### Boundary conditions and mesh types

- **Wedge (axisymmetric) patches are untested.** They are uncoupled patches with
  non-zero size, so they go down the generic
  `valueInternalCoeffs`/`valueBoundaryCoeffs` path; the component-mixing
  transform that a wedge applies is **not** accounted for in the implicit
  pressure-gradient block of the momentum rows. They are at least zero-flux
  correct in the continuity row and the flux reconstruction, via the
  `zeroFluxPatch` rule. No axisymmetric case has been run.
- **`symmetry`, `symmetryPlane` and `slip` patches** are correct by construction
  in the continuity row and the reconstructed flux, and have been exercised on
  the 3-D cases; their momentum-row treatment uses the same generic transform
  path as a wedge and has not been verified systematically.
- **Cyclic patches are untested.** They are handled by the same code path as
  processor patches - `syncTools::swapBoundaryCellList` returns the far-side
  value for both - so support follows structurally, but no case in this
  repository exercises a cyclic patch, and the parallel-consistency evidence of
  section 4 is entirely on processor patches.
- **Non-conformal and stitched (`nonConformalCoupled`) patches are not
  handled.**
- **Mesh coverage is structured hexahedral plus one unstructured
  snappyHexMesh mesh.** Polyhedral and tetrahedral meshes are untested. The
  non-orthogonal correction is exercised on the snappyHexMesh case (80.9 degrees
  maximum non-orthogonality, 5,408 skew faces), and not more broadly than that.

### Assembled terms that are absent

- **No `fvModels`, no `fvConstraints`.** `fvOptions`-style momentum sources,
  porous zones and turbulence sources are unsupported and silently not
  assembled.
- **No MRF, no mesh motion, no dynamic mesh.**
- **No `adjustPhi`.** A closed or fixed-flux domain with an inconsistent
  inflow/outflow balance produces a singular continuity block; the
  `adjustPhi`-style flux scaling that the segregated solver applies is not
  performed.
- **The COO sparsity pattern is assumed static** across outer iterations. A
  boundary condition that switches between fixed value and zero gradient
  mid-run, or any topology change, breaks the caching in both backends.

### Backends, solver settings and diagnostics

- **The Eigen backend is serial only** and raises a fatal error in a parallel
  run. `SparseLU` memory growth makes it uneconomical in 3-D.
- **No preconditioner is universal.** The shipped default is the only one
  measured to be both fast and robust on every topology tested, but the
  selection remains a per-case decision made through `petscOptions`, and a new
  topology is a new experiment.
- **`relTol` cannot be tuned once and forgotten.** Four regimes have four
  different measured recommendations (small laminar, large laminar with
  pseudo-transient continuation, small RANS, large RANS). Getting it wrong
  either stalls the outer loop or makes each outer iteration intractable. See
  `doc/USAGE.md`.
- **`reusePreconditioner` is measurably counterproductive** and is off by
  default; the Picard operator changes too fast between outer iterations for a
  stale preconditioner to pay for itself.
- **The pseudo-transient length scale is `V^(1/3)`**, which under-estimates the
  in-plane cell size on thin 2-D meshes. Off by default.
- **The `Final residual` of a block whose reference response is numerically
  zero is meaningless** - for example `Uy` at the first iteration of a
  lid-driven cavity, where the OpenFOAM-style normalisation factor collapses and
  prints a value near 0.75 while the absolute linear residual is about 1e-15.
  Convergence control uses only the *initial* residuals, which are clean.
- **`residualControl` keys for segregated fields match on exact field names, not
  regular expressions.** `"(k|epsilon)" 1e-4;` is valid in the `SIMPLE`
  dictionary and silently never matches in the `COUPLED` dictionary.
- **The turbulence residual used for convergence control is the one measured at
  the end of the previous outer iteration**, lagged one iteration like the
  turbulence itself, so the reported converged iteration can be one later than a
  same-iteration test would give.

### Platform-scoped results

- **Every GPU conclusion is scoped to the hardware it was measured on**: one
  workstation-class card with 1/64-rate double precision, under WSL2, with no
  MPS available and a non-CUDA-aware MPI. On that platform the device path is
  numerically exact but not faster - it ties the CPU at best, is about 7 per
  cent slower at fixed preconditioner and rank count, is roughly 30 times slower
  on small problems, and scales *inversely* with rank count because each rank
  opens its own CUDA context on one device. None of that should be extrapolated
  to a datacentre part, to bare-metal Linux, or to a mesh large enough that the
  per-Picard matrix re-ship stops dominating.
- **The AmgX findings are version- and platform-scoped** to AmgX 2.4.0 as pinned
  by PETSc, under WSL2, with the host driver tested. Two local PETSc source
  changes were needed to get that far; they were **not** submitted upstream and
  are unverified against current PETSc.
- **Wall-clock competitiveness is not claimed.** At production tolerances on the
  483,000-cell 3-D case the best coupled configuration is about 2.2 times behind
  the segregated solver in wall clock while using 24 outer iterations against
  142. The coupled solver's demonstrated advantages are deep convergence that
  the segregated solver cannot reach on that mesh at any cost, zero-relaxation
  robustness, and mesh-independent iteration counts - not raw speed.

---

## 10. Re-running the verification

Nothing in this document requires special tooling. Build the solver, run the
tutorials, and compare fields with the script in `tools/`.

### The regression gate

```sh
. /opt/openfoam14/etc/bashrc     # path of the local OpenFOAM installation
./Allwmake
./Allrun
```

`./Allrun` runs every case under `tutorials/`, writes each case's console output
to `<case>/log.Allrun`, prints a pass/fail summary and exits non-zero if any
case failed. A case counts as failed when its own `Allrun` exits non-zero or
when `FOAM FATAL ERROR` appears in any of its logs. The four tutorials are a lid-driven cavity, a
2-D interrupted-fin strip channel in laminar and kOmegaSST form, and a 3-D
interrupted-fin duct.

### What to look for in a log

- `coupled assembly check: sum|div(phi)| = ...` on every outer iteration. This
  should sit at machine precision (order 1e-16 relative to the characteristic
  flux) from the first iteration onwards. Anything above about 1e-10 relative
  means either an assembly inconsistency or an inner solve that did not
  converge - check `relTol` before suspecting the assembly.
- `COUPLED solution converged in N iterations`. Compare `N` against section 4
  when changing rank count: it should not move at all in laminar cases.
- The pressure-level line at start-up, either
  `Pressure level: pinned at global cell 0 to p = 0` or
  `Pressure level: set by fixedValue patch <name>`. A case that should have a
  pressure outlet but reports pinning has a boundary-condition problem.

### Comparing two solutions

`tools/compare_fields.py` reads the ASCII `internalField` of `U` from two
case/time directories on a structured 2-D mesh and reports the largest cell-wise
difference anywhere in the domain plus the profile along the vertical
centreline:

```sh
python3 tools/compare_fields.py <caseA> <timeA> <caseB> <timeB> [n]
```

The same script serves all three comparisons in this document:

- **coupled against segregated** - run the case with `coupledSimpleFoam`, then
  the same case directory with `foamRun -solver incompressibleFluid`, then
  compare. **Tighten the reference first**: set the segregated
  `residualControl` to 1e-9 and raise `endTime` accordingly, or the difference
  you measure will be dominated by the reference's own convergence error
  (section 3).
- **parallel against serial** - `decomposePar`, `mpirun -np N coupledSimpleFoam
  -parallel`, `reconstructPar -latestTime`, then compare the reconstructed time
  against a serial run of the same case. Check the outer-iteration counts match
  exactly as well as the fields.
- **backend against backend** - run the same case once with
  `linearSolver SparseLU` and once with `linearSolver PETSc`, both serially, and
  compare. This separates a backend difference from a decomposition difference.

### Reproducing the Rhie-Chow attribution

To confirm that a residual difference against a segregated reference is the
damping-definition difference of section 3 rather than an error, set

```
COUPLED
{
    rhieChowFactor  0.7;    // the reference run's momentum relaxation factor
}
```

and re-run. If the difference collapses by roughly an order of magnitude at
essentially unchanged outer-iteration count, the discretisations agree and the
gap was the `D_f` definition. Restore the default of 1.0 afterwards; it is the
correct value for a solver with no relaxation factor.

### The cross-implementation check

```sh
cd tools/reference_prototype
python3 coupled_vs_simple.py                 # 240 x 48 mesh, Re = 400
python3 coupled_vs_simple.py --re 1000       # higher Reynolds number
python3 coupled_vs_simple.py --quick         # 120 x 24, for a fast check
```

Requires only Python 3 with NumPy and SciPy; no OpenFOAM environment. Iteration
counts, wall-clock times and pressure drop for each configuration are written to
`results_re<Re>_<nx>x<ny>.json`, which is what section 5 tabulates.

### A note on scripting parallel runs

`mpirun` drains standard input. A validation script piped into a shell loses
everything after the first `mpirun` unless every invocation is given
`< /dev/null`. A script that appears to stop silently after its first parallel step is almost
always this.
