# Validation

This document is the correctness evidence for `coupledSimpleFoam`: what has
been checked, by what method, what was measured, and - at least as important -
what has not been checked. Performance numbers live in `doc/BENCHMARKS.md`;
this file is only about whether the answers are right.

Sections 1 to 7 are the verification layers, section 8 is the defect history,
section 9 relates the observed behaviour to the published literature, section
10 is the explicit list of what is *not* covered, and section 11 explains how
to reproduce the whole set.

## Contents

1. [Verification strategy](#1-verification-strategy)
2. [Discrete mass conservation](#2-discrete-mass-conservation)
3. [Agreement with the segregated solver](#3-agreement-with-the-segregated-solver)
4. [Parallel consistency](#4-parallel-consistency)
5. [Cross-implementation check](#5-cross-implementation-check)
6. [Turbulence regression](#6-turbulence-regression)
7. [Energy transport](#7-energy-transport)
8. [Defects found and fixed during verification](#8-defects-found-and-fixed-during-verification)
9. [Comparison with the published literature](#9-comparison-with-the-published-literature)
10. [What is not validated](#10-what-is-not-validated)
11. [Re-running the verification](#11-re-running-the-verification)

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
several of these layers is described in section 8.1, and it survived exactly
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
  section 8.1 stayed invisible to this check while corrupting an integral
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

### Agreement under mesh refinement

If the residual difference between the two solvers is the order-`h^2` damping
term identified above rather than an implementation error, it has to shrink
under refinement instead of holding at a fixed level. That is a testable
prediction, and it has been tested on a single structured three-dimensional
interrupted-fin duct meshed at four resolutions, with both solvers run at their
best settings on ten ranks, identical schemes, and identical `residualControl`
(`U` 1e-5, `p` 1e-4). Every run formally converged.

| cells | `dp` difference | all-cell rms abs(dU), per cent of field maximum |
| --- | --- | --- |
| 269,000 | 0.18 % | 0.12 % |
| 533,000 | 0.12 % | 0.090 % |
| 1,070,000 | 0.063 % | 0.074 % |
| 2,150,400 | 0.014 % | 0.075 % |

The pressure-drop difference falls by more than a factor of ten across a factor
of eight in cell count. An assembly error does not behave that way; a term that
vanishes with the mesh does, and the rate is the one the `h^2` attribution
predicts. The field rms falls more slowly and settles near 0.075 per cent, which
supports the same attribution rather than contradicting it: the rms is dominated
by the fin corners, where `rAU` jumps hardest between neighbours, in the same
way that the cavity's all-cell difference is dominated by the singular lid
corner while its centreline difference is two orders of magnitude smaller.

The finest point additionally separates accuracy from solver effort. On the
2,150,400-cell mesh the coupled `dp` is 0.96255897 against the segregated
0.96269619, and over all 2,150,400 cells the rms velocity difference is 1.26e-3,
0.075 per cent of the field maximum. The same case solved slowly, with tight
inner tolerances throughout, differs from segregated by 1.28e-3 - the same
number. The gap to segregated is therefore a property of the discretisation and
not of how hard the linear systems were solved, so the tuned configuration of
`doc/BENCHMARKS.md` section 9.2, which reaches the answer in a fraction of the
wall clock, does not pay for that speed in accuracy. Normalised
`abs(div(phi))` on that run is about 1e-9 rather than the 1e-16 of the small
tightly solved cases; that is the loose inner tolerance showing up in the
readout of section 2, not an assembly inconsistency, and the field and integral
agreement improve at that mesh level rather than degrading.

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

## 7. Energy transport

`COUPLED/solveEnergy` adds a temperature equation

```
fvm::div(phi, T) - fvm::laplacian(alphaEff, T) = 0,    alphaEff = nu/Pr + nut/Prt
```

solved segregated after the coupled p-U solve, on the conservative Rhie-Chow
flux that solve produced. This section is the evidence for it. Everything below
is on `tutorials/heatedDuct` unless stated otherwise: a plane channel 10 m long
and 1 m high, 200 x 40 cells, Re = U*Dh/nu = 20, Pr = 7, Pe = 140, inlet at
300 K and both walls at 350 K.

Three checks, in increasing order of what they can catch:

1. **Against a segregated reference solving the identical discrete equation**,
   which catches an error in the equation or in the flux it is given.
2. **Against an exact analytical result**, Nu = 7.5407, which catches an error
   the reference shares - because the reference is OpenFOAM's own
   `scalarTransport`, not an independent discretisation.
3. **Discrete energy conservation**, which is the temperature counterpart of
   section 2 and catches a boundary term that is in the matrix but not in the
   balance, or the reverse.

### 7.1 Two segregated references, and why both

OpenFOAM 14's `scalarTransport` function object solves

```c++
fvm::ddt(s) + fvm::div(phi, s, "div(phi,s)")
  - fvm::laplacian(D, s, "laplacian(Ds,s)") == fvModels.source(s)
```

with `D = alphal*nu + alphat*nut` when `diffusivity viscosity` is selected.
With `ddtSchemes { default steadyState; }`, `alphal = 1/Pr` and
`alphat = 1/Prt`, that is the solver's `TEqn` **term for term**: same operators,
same scheme lookups (`div(phi,T)` in both; the laplacian resolves through the
`laplacianSchemes` `default` in both, the solver naming its diffusivity
`alphaEff` and the function object naming its own `DT`), and the same
`relax(0)`-is-a-no-op relaxation path. It is therefore the right reference, and
it is used two ways.

**Reference A - frozen flux.** The converged `U`, `p` and `phi` of the coupled
run are copied into a fresh case and `foamRun -solver functions` with
`subSolver incompressibleFluid` executes only the function object. The flux is
then *bit-identical* to the one the coupled solver used, so the two are solving
the same linear system and any difference is linear-solver tolerance and
nothing else. This is the check that isolates the energy equation from the
flow.

**Reference B - full segregated run.** `foamRun -solver incompressibleFluid`
solves the flow with SIMPLE from the same `0/` fields and schemes, with the
function object solving `T` alongside it. Here the flux differs by the
Rhie-Chow damping definition of section 3, so the temperature difference is a
discretisation difference, not an error, and it has to be read with the same
caution section 3 sets out - including that the reference's own convergence
level dominates at production criteria.

Measured, over all 8 000 cells, on a field whose range is 49.28 K:

| Comparison | max abs(dT) | rms abs(dT) | wall heat flux difference |
| --- | --- | --- | --- |
| coupled vs reference A (frozen flux) | 4.05e-05 K | 2.78e-06 K | 1.87e-08, 1.04e-08 relative |
| coupled vs reference B, `p 1e-4 / U 1e-5` | 4.52e-02 K | 3.40e-03 K | 2.20e-04, 1.22e-04 relative |
| coupled vs reference B, `p 1e-9 / U 1e-9` | 4.52e-02 K | 3.04e-03 K | 1.30e-04, 7.26e-05 relative |
| reference B production vs reference B tight | 3.68e-03 K | 9.43e-04 K | 8.95e-05, 4.99e-05 relative |

Reference A is the load-bearing row. The two agree to **8.2e-07 of the field
range**, and the residual difference is not a discretisation difference at all
but the coupled run's own convergence level: its temperature equation is solved
once per outer iteration and the loop stops at `residualControl T 1e-6`, while
the reference was iterated to a `T` residual of 9.5e-13 and `No Iterations 0`.
Tightening the coupled criterion moves this figure and nothing else. The wall
heat fluxes agree to 1.0e-08 relative and the Nusselt numbers to 1e-09, which
is the statement that matters: **given the same flux, the two implementations
produce the same discrete temperature field.**

One practical point that row also establishes. With a second-order upwind
`div(phi,T)` the convection term carries a deferred correction, so the
temperature equation is not linear in one pass even at frozen flux: the
reference needed roughly thirty passes to take its residual from 3.7e-03 to
9.5e-13. Five passes - the obvious choice - leaves it at 6.8e-05 and the
comparison then reads 1.1e-02 K, which is the reference's convergence and not
the solver's error. A frozen-flux check has to converge the correction loop
before it means anything.

Reference B is a discretisation comparison and behaves like section 3's. The
coupled-versus-segregated difference, 4.5e-02 K, is an order of magnitude
larger than the reference's own convergence error, 3.7e-03 K, so it is a real
difference between the two discretisations rather than an artefact of stopping
the reference early - and it is the Rhie-Chow damping difference of section 3
arriving in the temperature field through `phi`, not anything belonging to the
energy equation. The inlet pressure drops carry the same signature: 13.0918
coupled against 13.1136 for the tightly converged segregated run, 0.17 %, which
is the middle of the 0.17-0.38 % band section 3 reports for the laminar 2-D
suite. The wall heat flux differs by 0.007 %.

### 7.2 The physics gate: Nu = 7.5407

Fully developed laminar flow between parallel plates held at a constant and
equal temperature has a Nusselt number of 7.5407 (Shah & London; see
`doc/REFERENCES.md`). The value is exact, it is independent of Pr and Re, and
it depends on the temperature field alone - both the conductivity and `rho*cp`
cancel out of

```
Nu = |dT/dy|_wall * Dh / (Tw - Tb),      Dh = 2*H
```

so reproducing it tests the discrete temperature solution and nothing else.
`tools/heated_duct_nusselt.py` computes it from the written `T` and `U` and the
cell centres of `foamPostProcess -func writeCellCentres`; the bulk temperature
is the flux-weighted mean over the cross-section and the wall gradient is
one-sided from the near-wall cell.

**The post-processor was calibrated first, so that its error is not charged to
the solver.** The fully developed profile is the first Graetz eigenmode: the
solution of `phi'' + beta*u(y)*phi = 0` with `phi(0) = phi(H) = 0` and
`u = 6*y*(H-y)/H^2`. Integrating that by shooting to a tolerance of 1e-12 gives
`beta = 7.5407008741` and, from the eigenfunction itself,
`Nu = |phi'(0)|*Dh/|phi_bulk| = 7.540690` - the published value to five
figures, which confirms the analytical target independently of any code.
Sampling that exact `phi` at the 200 x 40 cell centres of this tutorial and
handing it to `heated_duct_nusselt.py` returns **7.5430, i.e. +0.030 %**. That
is the post-processor's own error at this mesh - the one-sided wall gradient
and the midpoint bulk integral, no solver involved - and it is therefore the
value the solver should be measured against: a computed Nusselt number near
7.543 means the discrete temperature field is right, and the residual 0.03 %
is the post-processing, not the physics.

On the tutorial, with the wall at 350 K and the inlet at 300 K:

| x, m | Tb, K | Tw - Tb, K | abs(dT/dy) at the wall | Nu | Nu/7.5407 |
| --- | --- | --- | --- | --- | --- |
| 1.975 | 313.811 | 36.189 | 143.928 | 7.9542 | 1.0548 |
| 2.975 | 317.615 | 32.385 | 124.352 | 7.6796 | 1.0184 |
| 3.975 | 320.952 | 29.048 | 110.203 | 7.5878 | 1.0062 |
| 4.975 | 323.926 | 26.074 | 98.521 | 7.5570 | 1.0022 |
| 5.975 | 326.589 | 23.411 | 88.338 | 7.5466 | 1.0008 |
| 6.975 | 328.978 | 21.022 | 79.286 | 7.5431 | 1.0003 |
| 7.975 | 331.123 | 18.877 | 71.186 | 7.5420 | 1.0002 |
| **8.975** | 333.049 | 16.951 | 63.920 | **7.5416** | **1.0001** |
| 9.975 | 334.800 | 15.200 | 57.465 | 7.5611 | 1.0027 |

**Nu = 7.5416 against the exact 7.5407: +0.012 %.** The approach is monotonic
from above through the thermal entry region and flattens from x = 6 m onward,
which is the behaviour the Graetz solution has and is itself evidence that the
station is in the developed region rather than a coincidence at one point.

Two features of the table are worth naming rather than glossing over. The last
station, x = 9.975 m, reads 7.5611 and is the only one that breaks the trend:
`zeroGradient` on `T` at the outlet forces the streamwise gradient to zero in
the last column, where physically it is still positive, and that perturbs the
last few cells. It is an artefact of the outlet condition, not of the energy
equation, and it is why the tutorial quotes x = 8.975 m. And the computed
7.5416 sits *below* the 7.5430 that the post-processor returns on the exact
field, so the solver's own discretisation error and the post-processor's are of
opposite sign and partly cancel; the honest reading is that both are of order
0.03 % and the agreement with 7.5407 to 0.012 % is not better than either
error separately.

Repeating the case at `residualControl U 1e-10 / p 1e-10 / T 1e-12` - 28 outer
iterations instead of 11 - gives 7.5416 at the same station, to four decimal
places. The result is converged, not a stopping artefact.

The case is sized so that the three assumptions behind the benchmark all hold
at once, and this is the part of the exercise that is easy to get wrong rather
than the implementation: the flow must be hydrodynamically developed (entry
length 0.05*Re*Dh = 2 m, against stations at 6 m and beyond); axial conduction
must be negligible (the ratio to axial convection is about 7.54/Pe_H^2 = 0.15 %
at Pe_H = u*H/alpha = 70, which is why Pr is 7 and not 0.71 - at Pe = 14 the
asymptote moves towards the low-Peclet limit of 8.117 and the case would
reproduce the wrong number for the right reason); and `Tw - Tb` must still be
resolvable at the measuring station, which it is, at about a quarter of its
inlet value at the outlet.

The constant-heat-flux counterpart is a second, independent analytical value.
For parallel plates with equal constant flux on both walls the fully developed
profile has a closed form,

```
T(y) - Tw = (C/alpha)*(y^3 - y^4/2 - y/2),    C = dTb/dx = 2*alpha*g/(u*H)
```

whose Nusselt number is exactly 8.235294. The same calibration applies: that
profile sampled on this mesh and passed through the post-processor gives
8.2377, +0.033 %. Setting the wall condition to `fixedGradient` with
`gradient 175` and changing nothing else gives **8.2466 at x = 9.025 m**, 0.11 % above the calibrated 8.2377 and 0.14 % above the exact 8.2353. The station matters more here than in the isothermal case: the wall temperature rises linearly for ever under a constant flux, so the `zeroGradient` outlet is a genuine perturbation over the last few columns (8.2828 at x = 9.975) and the entry region is longer (8.7484 at x = 3.025, 8.3615 at 5.025, 8.2695 at 7.025). The approach is monotonic from above and lands on the calibrated value. The bulk temperature
gradient is a third check with a closed form, `dTb/dx = 2*alpha*g/(u*H)`,
which for this case is exactly 5 K/m: measured 4.99963 K/m between x = 5.025 and x = 9.025, -0.007 %. Taking the last station instead gives 5.0105, +0.21 %, which is the same outlet effect and not a different result.

### 7.3 Discrete energy conservation

Every outer iteration prints

```
coupled energy check: wall heat flux = 1.7944553015, enthalpy outflow = 1.73999043168, boundary diffusion = 1.73999043164, imbalance = 2.36837395898e-09 %
```

Summing the assembled equation over every cell telescopes the internal faces
away and leaves exactly two boundary sums - the convective outflow
`sum(phi_b*T_b)` and the diffusive influx
`sum(alphaEff_b*|Sf|_b*snGrad(T)_b)` - so on a converged solution they must be
equal. Both are taken with the boundary conditions' own gradient coefficient
functions, so they are the fluxes the matrix used rather than a reconstruction
of them, exactly as section 2's continuity check uses the same expression as
the assembly.

On the tutorial the imbalance sits between **1.5e-11 % and 2.4e-09 %** over
every one of the eleven outer iterations, with no drift: the individual values
are -1.5e-11, -1.8e-09, -1.8e-09, +1.2e-09, -1.3e-09, -1.6e-10, -2.2e-09,
+8.2e-10, +9.9e-10, +2.4e-09 %. It is a linear-solver tolerance and not a
physical residual - the `T` equation is solved to `tolerance 1e-12` on a field
of order 300 K, i.e. about 1e-14 relative, against boundary sums of order 1.7 -
and it changes sign from iteration to iteration as roundoff does.

Two of the boundary-condition cases in 7.4 close it more sharply still, because
their answers are exactly representable: with adiabatic walls the wall heat
flux is **exactly 0** and with `fixedGradient 175` on both walls it is
**exactly 2.5 K m3/s**, which is `2*alpha*g*A_wall = 2*(0.1/7)*175*0.5`, to
every printed digit.

This is the energy analogue of section 2, and it has the same two legitimate
exceptions, plus one that is specific to a ratio: the imbalance is meaningless
while the run is far from converged; on a non-orthogonal mesh the explicit
non-orthogonal correction of the laplacian sits outside the identity, so the
floor there is the size of that correction rather than roundoff; and when both
sums are themselves zero - the adiabatic case, where they are 2.7e-14 and
2.6e-15 - the *percentage* is a ratio of two roundoff-level numbers and means
nothing, while the absolute values it is formed from are the meaningful
statement.

### 7.4 Boundary conditions

Each of the wall conditions in `doc/USAGE.md` 9.4 was run on this case and
checked against something that does not depend on the implementation.

| Condition | Check | Result |
| --- | --- | --- |
| `fixedValue` (isothermal) | Nu = 7.5407 | **7.5416, +0.012 %** (7.2) |
| `zeroGradient` (adiabatic) | `T` must stay exactly at the inlet value everywhere, and the wall heat flux exactly zero | `T` is **exactly** 300 K in all 8 000 cells - `max abs(T - 300) = 0.000e+00`, not merely small - and the wall heat flux is exactly 0. This is the discrete conservation statement in its sharpest form: a constant field with constant boundary values is an exact solution of the assembled equation only if the flux the convection term is built on is divergence free cell by cell |
| `fixedGradient` (prescribed flux) | Nu = 8.235 and the closed-form bulk gradient | wall heat flux exactly 2.5 K m3/s, the analytical `2*alpha*g*A_wall`; energy imbalance 6.4e-11 %; Nu = 8.2466 against a calibrated 8.2377; bulk gradient -0.007 % from the closed form |
| `symmetry` | a half-channel with a symmetry plane on the centreline must reproduce the lower half of the full channel cell for cell | the half-channel wall heat flux is 0.897227658459 K m3/s against exactly half
the full channel's 1.79445531882, i.e. **1.1e-09 relative** - the symmetry
plane carries exactly zero heat flux. The temperature fields agree to
max 1.77e-05 K and rms 2.40e-06 K over the 4 000 shared cells |

The symmetry row needs one qualification, and it is about the flow rather than
the energy equation. The half-channel and full-channel velocity fields differ
by up to 2.55e-05 in `Ux`, and that figure barely moves when both runs are
tightened from `U 1e-5` to `U 1e-10`, so it is not a convergence artefact: the
momentum diagonal `aP` picks up the symmetry patch's `internalCoeffs` in the
half mesh where the full mesh has an off-diagonal link to the mirror cell.
Those are equivalent in the residual but not in `aP`, and `aP` sets the
Rhie-Chow coefficient `rAU`, so the damping differs by O(h) in the single row
of cells next to the plane. That is the same class of difference section 3
attributes to the Rhie-Chow definition, it predates the energy equation, and it
is what the remaining 1.77e-05 K in `T` is inherited from. The energy
equation's own contribution to this comparison - zero flux through the
constraint patch - is the 1.1e-09 figure above.

`externalWallHeatFluxTemperature` is **not usable** and this was confirmed by
running it: it belongs to the thermophysical transport framework and looks up a
`kappaEff` that a passive-scalar run does not register, so it fails at start-up.
`doc/USAGE.md` 9.4 gives the `fixedGradient` and `mixed` equivalents of its two
useful modes. `wedge` (axisymmetric) patches remain untested for the energy
equation as they are for the flow (section 10).

### 7.5 Parallel

The energy step has two pieces of parallel-specific code: the balance sums skip
`coupled()` patches, because processor and cyclic faces are internal faces of
the global mesh and telescope away like any other, and the three sums are then
reduced across ranks. Both are checked the same way section 4 checks the flow -
by running the same case serially and under MPI and comparing cell by cell.

On `heatedDuct` at 4 ranks (`simple` decomposition, `4 1 1`, PETSc backend):

| Quantity | Serial | 4 ranks |
| --- | --- | --- |
| outer iterations | 11 | 11 |
| wall heat flux, K m3/s | 1.7944553015 | 1.7944554468 |
| max abs(dT) over all 8 000 cells | - | 1.00e-09 K |
| rms abs(dT) | - | 3.55e-10 K |
| wall heat flux difference | - | 2.0e-11, 1.1e-11 relative |

The iteration count is identical, not within one, which is the requirement
section 4 sets. The 1e-09 K field agreement is at the floor set by
`writePrecision 12` on values of order 300, so it is a reproduction rather than
an approximation.

### 7.6 What this does not establish

- **The flow is unaffected, and that is verified rather than argued.** With
  `solveEnergy` absent and with `solveEnergy no` written explicitly, the four
  existing tutorials produce logs that are identical line for line to the
  pre-change build, `stripChannel`'s inlet pressure drop included to every
  printed digit. That is the evidence that the passive-scalar claim holds in
  the code and not only on paper.
- **Only laminar heat transfer is validated.** No RANS case with
  `solveEnergy` has been run. The turbulent heat flux is the constant-`Prt`
  gradient diffusion hypothesis with no thermal wall function, and on a
  wall-function mesh that is a known systematic error rather than a small one.
- **No buoyancy, and nothing detects its absence.** The passive-scalar
  structure is exact only while the flow does not depend on the temperature.
  A case where it does will converge and produce a wrong answer silently.
- **One mesh, one topology, one Prandtl number.** The energy equation has been
  exercised on a single orthogonal structured mesh. Non-orthogonal meshes,
  unstructured meshes and the interrupted-fin topologies of the other
  tutorials have not been run with `solveEnergy` on.
- The comparison in 7.1 is against OpenFOAM's own function object, which
  shares OpenFOAM's discretisation. It establishes that the solver assembles
  and solves the intended equation on the intended flux; the independent
  statement about whether that equation is right is 7.2.

---

## 8. Defects found and fixed during verification

Two assembly defects and one configuration defect were found and corrected
during verification. They are recorded here because each one is a trap a
reimplementation would fall into, and because the way each was detected says
something about which check is load-bearing.

### 8.1 Constraint-patch mass leak on non-axis-aligned faces

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

### 8.2 Patch diagonal counted twice

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

### 8.3 A linear-solver default validated on one mesh topology only

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

## 9. Comparison with the published literature

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
   cell, and 29 against 555 on the 2,150,400-cell three-dimensional duct).
2. *Outer-iteration counts that are largely mesh-independent.* Measured at 12
   outer iterations on cavities of 4,096, 16,384 and 65,536 cells - a 16-fold
   range - while SIMPLE goes from 797 to 2,400 over the first factor of four
   alone. The reference prototype shows the same signature independently
   (149 -> 284 for SIMPLE under refinement, 27 to 31 for coupled). The same
   signature holds in three dimensions at production size: over the four-point
   refinement of section 3 the coupled count moves 21 -> 24 -> 25 -> 29 while
   the segregated count on the identical meshes moves 221 -> 296 -> 404 -> 555.
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

The positive half of the same position has since been measured, at 8.6 million
unknowns, and it holds without being economical. Giving hypre block
interpolation and a nodal Schwarz smoother on that matrix improves the residual
after three outer iterations sixteen-fold over plain nodal coarsening, so block
awareness does what the literature says it does; but it costs 363 to 370 seconds
per outer iteration against 98.7 for the Schur splitting, the nodal systems-AMG
variants are slower still (389 s and 684 s) and barely move the residual at all,
and GAMG with unsmoothed aggregation and a block Jacobi smoother diverges with
NaN on its first application. Diagonal equation scaling lifts the best block
configuration to 292 s while degrading the adopted Schur configuration from
98.7 s to 196.7 s. Monolithic block AMG is therefore measured and rejected on
this system rather than outstanding: assembled from generic library components
it costs three to four times more per iteration than the Schur splitting, and a
competitive one needs purpose-built coarsening, interpolation and smoothers.
Those measurements are only meaningful because every run passed
`-pc_hypre_boomeramg_numfunctions 4`. PETSc does not forward the matrix block
size to hypre - `MatSetBlockSize(A,4)` only sets the default value of an option
whose hypre call is guarded by a flag test - so without that option hypre runs
at `num_functions = 1` and every systems-AMG setting silently degenerates to the
scalar case that was already known to fail. Every run also carried
`-options_left`, which reported no unused options.

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

## 10. What is not validated

This section is deliberately exhaustive. Anything not listed in sections 2 to 7
should be assumed untested.

### Physics and equation scope

- **Steady only.** There is no `fvm::ddt` term; cases must use
  `ddtSchemes { default steadyState; }`. Pseudo-transient continuation is a
  relaxation device, not a time-accurate scheme, and nothing about it has been
  validated as time accuracy.
- **Incompressible, single-phase, constant density.** The pressure is
  kinematic. No compressible support, no multiphase, no free surface.
- **Heat transfer is a passive scalar only** (section 7). Temperature is
  transported by the flow and never acts on it, so **buoyancy is absent**:
  natural and mixed convection are outside the model and nothing in the run
  will say so. Properties are constant - no temperature-dependent viscosity,
  density or conductivity - and there is no viscous heating, no compression
  work, no conjugate heat transfer with a solid region, no radiation and no
  volumetric source. The turbulent heat flux is the constant-`Prt` gradient
  diffusion hypothesis with **no thermal wall function**, and no RANS heat
  transfer case has been run at all: section 7 is entirely laminar.
- **No formal order-of-accuracy or grid-convergence study.** The discretisation
  is OpenFOAM's own, and the verification strategy is equivalence to the
  segregated solver rather than an independent demonstration of formal order.
  No Richardson extrapolation or mesh-refinement error study has been performed.
  The four-point refinement of section 3 does not close this gap: it measures
  how the *difference between the two solvers* behaves under refinement, both of
  them carrying the same discretisation error, and is evidence for the Rhie-Chow
  attribution rather than a statement about either solver's error against an
  exact or extrapolated solution.
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
  topology is a new experiment. The default is a robustness choice and is paid
  for in wall clock at size: the `selfp` Schur approximation rebuilds an
  explicit sparse matrix product every outer iteration and costs 555 s per
  outer iteration at 8.6 million unknowns, against 98.7 s for
  `-pc_fieldsplit_schur_precondition a11`, which reuses the pressure block the
  assembly already builds. On small or highly non-orthogonal meshes `selfp`
  remains the more robust of the two, which is why it stays the default.
- **`relTol` cannot be tuned once and forgotten.** Four regimes have four
  different measured recommendations (small laminar, large laminar with
  pseudo-transient continuation, small RANS, large RANS). Getting it wrong
  either stalls the outer loop or makes each outer iteration intractable. See
  `doc/USAGE.md`. `adaptiveTolerance` removes the choice on large
  three-dimensional cases but replaces it with `relTolMax`, which carries the
  same regime dependence: 3e-2 is the measured value on the large
  three-dimensional benchmark and is too loose on small two-dimensional ones.
- **`reusePreconditioner` is measurably counterproductive** on the cases where
  it was measured and is off by default; freezing the preconditioner after the
  first build does not pay, because the Picard operator changes too fast between
  outer iterations for a stale preconditioner to be worth its reduced quality.
  The separate `pcReuseInterval` path, which rebuilds every N-th outer iteration
  rather than never, does pay on large three-dimensional cases with a cheap
  hierarchy, and the two are not interchangeable: when both are set the older
  freeze wins.
- **`adaptiveTolerance` and `pcReuseInterval` are regression tested, not
  independently validated at scale.** Both are off by default and both have been
  regression tested against the shipped tutorials, which is what establishes
  that leaving them off reproduces the previous logs byte for byte and that
  switching them on does not move the converged answer of those cases. Beyond
  that, the evidence for them is a single benchmark geometry - the
  2,150,400-cell duct and its four-point refinement - on one machine at one rank
  count. Neither the Eisenstat-Walker forcing sequence nor the reuse interval
  has been exercised on a second large three-dimensional topology, on a
  non-orthogonal 3-D mesh, or under RANS at that size. The tuned settings are a
  measured starting point, not a validated default, which is why they are not
  the defaults.
- **The adaptive path changes the inner convergence test, and that change is
  covered by the same evidence.** An Eisenstat-Walker tolerance is defined
  against the current iterate, so the adaptive path switches the KSP test to the
  initial-residual norm (`KSPConvergedDefaultSetUIRNorm`); under PETSc's default
  test relative to `abs(b)`, a warm-started loose solve returns after zero
  Krylov iterations and the outer loop stalls while every printed diagnostic
  looks healthy. The fixed-`relTol` path is unchanged and keeps the default
  test.
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

- **Where wall clock is concerned, the crossover is what is scoped, not the
  answer.** The earlier measurement on the 483,000-cell 3-D case, with the
  configuration and defaults of the time, put the best coupled run about 2.2
  times behind the segregated solver in wall clock while using 24 outer
  iterations against 142. That figure stands as a record of that case at that
  stage of tuning; it is not the current position. On the 2,150,400-cell
  three-dimensional duct the tuned configuration of `doc/BENCHMARKS.md`
  section 9.2 converges in 203 s against the segregated solver's 478 s to the
  same criteria - 2.3 times faster, with the accuracy evidence of section 3 -
  and the four-point sweep of section 3 puts the crossover near 300,000 cells.
  What that crossover is scoped to is one ten-core workstation, ten ranks, one
  laminar three-dimensional topology and these two configurations: a different
  machine, a different topology, a RANS case, or an untuned coupled
  configuration will move it, and below the crossover the segregated solver
  remains the faster of the two. The advantages that do not depend on size are
  unchanged - deep convergence the segregated solver cannot reach on the
  483,000-cell mesh at any cost, zero-relaxation robustness, and
  mesh-independent iteration counts.

---

## 11. Re-running the verification

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
when `FOAM FATAL ERROR` appears in any of its logs. The five tutorials are a lid-driven cavity, a
2-D interrupted-fin strip channel in laminar and kOmegaSST form, and a 3-D
interrupted-fin duct.

### What to look for in a log

- `coupled assembly check: sum|div(phi)| = ...` on every outer iteration. This
  should sit at machine precision (order 1e-16 relative to the characteristic
  flux) from the first iteration onwards. Anything above about 1e-10 relative
  means either an assembly inconsistency or an inner solve that did not
  converge - check `relTol` before suspecting the assembly. A configuration that
  deliberately solves the inner systems loosely reads higher for that reason
  alone: the tuned large-case settings of `doc/BENCHMARKS.md` section 9.2 sit
  at about 1e-9 normalised while agreeing with the segregated solver to 0.014
  per cent on `dp`.
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
