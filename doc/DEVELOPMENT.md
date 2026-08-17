# Development

Contributor and maintainer guide: how the code is organised, which parts are
load-bearing, and what has to be verified before a change is accepted.

This document assumes the solver is already built and runs. For prerequisites
and build options see `doc/INSTALL.md`; for the `COUPLED` dictionary and the
user-visible behaviour see `doc/USAGE.md`; for the measurements quoted here in
full see `doc/BENCHMARKS.md`; for the pull-request checklist see
`CONTRIBUTING.md`.

Contents:

1. [Code map](#1-code-map)
2. [The backend abstraction](#2-the-backend-abstraction)
3. [OpenFOAM 14 API notes](#3-openfoam-14-api-notes)
4. [Build system](#4-build-system)
5. [Testing and regression protocol](#5-testing-and-regression-protocol)
6. [Extension roadmap](#6-extension-roadmap)
7. [Coding conventions](#7-coding-conventions)


## 1. Code map

The solver is four translation units and two headers-as-include-files, all
under `src/coupledSimpleFoam/`. It is deliberately small: there is one
algorithm, and it is written out in one place rather than distributed across a
class hierarchy.

| File | Lines | Responsibility |
| --- | --- | --- |
| `coupledSimpleFoam.C` | ~1170 | The whole algorithm: controls, global indexing, assembly, solve, flux reconstruction, residuals, turbulence update |
| `createFields.H` | ~105 | Constructs `p`, `U`, `phi`, the viscosity model, the momentum-transport model and `rAU` |
| `coupledLinearSystem.H` / `.C` | ~215 / ~200 | `Foam::coupledLinearSystem` (COO triplet store) and `Foam::coupledSolver` (backend interface plus the selection factory) |
| `eigenCoupledSolver.H` / `.C` | ~90 / ~210 | Serial Eigen backend: `SparseLU` (direct) and `BiCGSTAB` + `IncompleteLUT` |
| `petscCoupledSolver.H` / `.C` | ~130 / ~350 | MPI backend: PETSc `MATMPIAIJ` + `KSP`, COO assembly, optional CUDA |
| `Make/files`, `Make/options` | - | Build description and backend discovery |

Reading order for a newcomer: the `Description` block at the top of
`coupledSimpleFoam.C` (it is the specification of the discretisation, in six
numbered steps), then `coupledLinearSystem.H` (the data structure everything
funnels into), then the outer loop in `coupledSimpleFoam.C`. The independent
Python implementation in `tools/reference_prototype/coupled_vs_simple.py` is
the same algorithm in 2-D with the same variable names, and is usually faster
to read than the C++ when the question is "what is this term".

### 1.1 What lives in `coupledSimpleFoam.C`

Everything before the outer loop is set up that never changes:

- **Controls.** `coupledDict` is a *copy* of the `COUPLED` sub-dictionary of
  `system/fvSolution`, taken once, falling back to `dictionary::null` when the
  sub-dictionary is absent (which is why every key has a default and the
  solver runs with no configuration at all). Note that the copy means
  dictionary lookups performed inside the loop, such as `pseudoUref`, still
  see start-up values: no control is runtime-modifiable.
- **Global indexing.** `globalIndex globalCells(mesh.nCells())`, from which
  `blockSize` (4), `rowStart`, `nLocalRows`, `nGlobalRows` and the per-cell
  global index array `gCell` are derived. `nbrGCell` is filled by
  `syncTools::swapBoundaryCellList` and holds, for every boundary face, the
  global cell index on the far side (meaningful on coupled patches).
- **`zeroFluxPatch(patchi)`** - lambda predicate identifying impermeable
  constraint patches (`symmetryFvPatch`, `symmetryPlaneFvPatch`,
  `wedgeFvPatch`, and a `slipFvPatchVectorField` on `U`). These are treated
  like walls in the continuity row and in the flux reconstruction. Section 3.3
  explains why the generic boundary path is not sufficient for them.
- **Pressure level.** Scans `p.boundaryField()` for a patch reporting
  `fixesValue()`, reduces the flag across ranks, and decides between "the
  boundary sets the level" and "pin the continuity row of global cell
  `pRefCell`". The chosen mode is echoed at start-up.
- **Empty directions.** `solveComp[c]` is false where `mesh.solutionD()[c]`
  is `-1`; those momentum rows become identity rows.
- **`Seff`** - a `surfaceVectorField` holding `|Sf| * Delta * d`, the effective
  area vector of the lagged Rhie-Chow term, built once from
  `mesh.nonOrthDeltaCoeffs()` and the owner-to-neighbour vector (boundary
  faces use `fvPatch::delta()`, and `fvPatch::deltaCoeffs()` rather than
  `nonOrthDeltaCoeffs` on uncoupled patches). On an orthogonal mesh
  `Seff == Sf`; in general it is the exact complement of the implicit
  orthogonal Laplacian, so the non-orthogonal correction stays explicit
  exactly as in a segregated `pEqn`.
- **Linear algebra.** One `coupledLinearSystem sys` and one
  `autoPtr<coupledSolver> lsolver`, both created once and reused.

Three lambdas carry the per-iteration bookkeeping:

- **`packX(xv)`** - writes the current `U` and `p` into the interleaved local
  solution vector.
- **`buildRemote()`** - fills `xRemote`, a `Map<scalar>` of solution values
  for columns owned by other ranks, keyed by global row index. It must be
  called before every residual evaluation, because `multiply()` raises a fatal
  error on a column it has no value for.
- **`blockResiduals(xv, res)`** - the OpenFOAM-style normalised initial
  residual, per block component:

  ```
  res[c] = sum_i |b_i - (A x)_i|
         / max( sum_i ( |(A x)_i - (A xRef)_i| + |b_i - (A xRef)_i| ), vSmall )
  ```

  summed over the rows of component `c`, with `xRef` the component-wise global
  average. This is the block generalisation of
  `lduMatrix::solver::normFactor`; all sums are reduced across ranks. Keeping
  the same normalisation as OpenFOAM is what makes `residualControl` mean the
  same thing here as in `SIMPLE`.

### 1.2 One outer iteration, end to end

The loop is `while (iter < nOuterIterations)`. `runTime` is incremented for
the time-directory name and `runTime.run()` is never consulted, so `endTime`
in `controlDict` does not terminate the run - `nOuterIterations` does.

**Field state to assembled system.**

1. **Momentum operator.** An unrelaxed
   `fvVectorMatrix UEqn(fvm::div(phi, U) + momentumTransport->divDevSigma(U))`
   - the stock `incompressibleFluid` momentum predictor without `ddt`, MRF or
   `fvModels`, and with no equation relaxation, because a coupled solve has no
   relaxation factor.
2. **`aP`** = `UEqn.diag()` broadcast to all three components, plus
   `UEqn.internalCoeffs()` accumulated onto the patch face-cells. This
   reproduces what `fvMatrix::A()` computes, in the pre-division form the
   Rhie-Chow coefficient needs. `aP` is used **only** to form `Df`; the matrix
   itself receives the raw `diag()` in the cell loop and the `internalCoeffs`
   in the patch loop, so the two must never be added twice.
3. **Pseudo-transient diagonal** (optional). `dtauDiag[celli] = V/dtau` with
   `dtau = pseudoCo * cbrt(V) / max(|U|, 0.05*Uref)`; added to `aP` here and
   to the momentum diagonals and RHS during assembly.
4. **Rhie-Chow coefficient.** `rAU = rhieChowFactor * V / cmptAv(aP)`, then
   `rAU.correctBoundaryConditions()` (its `zeroGradient` patches make the
   patch value the owner value and exchange coupled patches), then
   `Df = fvc::interpolate(rAU)`.
5. **Lagged gradient.** `gradP = fvc::grad(p)` and
   `gradPf = fvc::interpolate(gradP)`, taken from the *previous* iteration's
   pressure. These exact fields are reused later in the flux reconstruction;
   that reuse is what makes the discrete continuity rows exact.
6. **Assembly.** `sys.clear()`, `sys.reserve(...)`, then five loops appending
   COO triplets with global indices:
   - cell loop: momentum diagonal (`diag + dtauDiag`), momentum source, the
     pseudo-transient RHS term, and identity rows for empty directions;
   - internal-face loop: momentum off-diagonals from `upper()` / `lower()`;
   - internal-face loop: the implicit Green-Gauss pressure gradient in the
     momentum rows, using `mesh.weights()` rather than 0.5;
   - internal-face loop: the continuity row - implicit velocity part, implicit
     orthogonal pressure Laplacian `Df*|Sf|*nonOrthDeltaCoeffs`, and the
     lagged `Df*(gradPf & Seff)` term on the RHS;
   - boundary-patch loop, which branches on `fvPatch::coupled()`. The coupled
     branch treats the face like an internal face whose neighbour lives
     elsewhere; the uncoupled branch uses the boundary conditions' own
     coefficient functions (section 3.3).
7. **Pressure pinning.** If no patch fixes `p`, `sys.setRowIdentity(pinRow,
   pRefValue)` replaces the *continuity* row of the reference cell, leaving
   its momentum rows untouched.

**Assembled system to solved fields.**

8. **Residual and convergence test.** `packX(xOld)`, `buildRemote()`,
   `blockResiduals(xOld, res0)`. The test is evaluated *before* the solve,
   against the system just assembled, and requires `iter > 1`. Segregated
   (turbulence) criteria measured at the end of the previous iteration are
   folded in here. On the converged iteration no solve happens, so the log
   prints `Final residual` equal to `Initial residual` with
   `No Iterations 0` - intended, not a stall.
9. **Solve.** `lsolver->solve(sys, xNew)` with `xNew` pre-loaded with `xOld`:
   the previous solution is the initial guess, which matters because the outer
   loop is a Picard iteration (see section 6.2).
10. **Unpack.** `U` and `p` are written back from `xNew`, then
    `correctBoundaryConditions()` on both, then `buildRemote()` and a second
    `blockResiduals` for the reported final residual.

**Solved fields to reconstructed flux.**

11. **Flux reconstruction.** `phi` is rebuilt face by face from exactly the
    same expression as the continuity row, with the same `Df`, the same
    `Seff` and the same *lagged* `gradPf`. Because the linear solve satisfied
    those rows, `div(phi)` is zero to machine precision. Coupled patches use
    remote `U` and `p` fetched by `syncTools::swapBoundaryCellList`;
    `zeroFluxPatch` patches are set to exactly zero.
12. **Continuity reporting.** `div(phi)` is accumulated per cell and reduced
    to produce the standard `time step continuity errors` line plus the
    diagnostic `coupled assembly check` line. That second line is a permanent
    verification print, not debug scaffolding; section 5.3 uses it as a gate.
13. **Turbulence update.** `viscosity->correct()` and
    `momentumTransport->correct()` run *after* the flux reconstruction, so the
    transport equations see the flux the coupled system just produced and
    `nut` is current for the next assembly. Their initial residuals are then
    read back through `convergenceControl` for the next iteration's
    convergence test - which is why the turbulence criteria are lagged one
    iteration behind the p-U criteria.
14. `runTime.write()`.

The gradient is refreshed at step 5 of the *next* iteration, never between the
matrix and the flux. Any change that breaks that pairing shows up immediately
as a loss of machine-precision continuity (section 5.3).


## 2. The backend abstraction

### 2.1 The COO triplet system

`Foam::coupledLinearSystem` is a coordinate-format (triplet) sparse matrix
with a right-hand side. It holds three `DynamicList`s (`rows_`, `cols_`,
`vals_`) and a `scalarField rhs_` sized `nLocalRows`.

Row and column indices are **global** and interleaved:

```
index = blockSize*globalCellIndex + component        component: u,v,w,p -> 0..3
```

Two ownership rules define the parallel contract:

- **A rank appends only triplets whose ROW it owns.** `rhs_` is indexed
  `gRow - rowStart_`, so an out-of-range row is a programming error, not a
  communication.
- **Columns may be remote.** They are reached through coupled patches, and the
  far-side global index comes from `nbrGCell`. No rank ever needs to know the
  full sparsity pattern.

Duplicate `(row, col)` pairs are legal and are summed by the backends
(`Eigen::SparseMatrix::setFromTriplets` and PETSc's COO path both do this
natively). The assembly therefore never searches for an existing entry, never
sorts, and never needs the final pattern in advance - it appends.

The remaining member functions:

- `clear()` / `reserve(n)` - per-iteration reset. The reserve heuristic is
  `blockSize*blockSize*(nCells + 2*nInternalFaces)`; it is an estimate, and if
  it is exceeded the `DynamicList`s simply grow at the cost of a reallocation.
- `setRowIdentity(gRow, value)` - zeroes the values of every triplet in that
  row in place and appends `(gRow, gRow, 1.0)`. The zeroed entries stay in the
  arrays as structural zeros. That is deliberate: it keeps the number and
  order of triplets identical on every outer iteration, which both backends
  depend on. The linear scan is O(nEntries) once per iteration.
- `multiply(xLocal, xRemote, result)` and `multiplyUniform(xRef, result)` -
  used only by the residual machinery.
- `rowData()` / `colData()` / `valData()` - raw contiguous arrays for batched
  backends. This is the interface that makes the PETSc COO path (and hence the
  GPU path) a direct handoff with no repacking.

### 2.2 The solver interface

`Foam::coupledSolver` is three functions:

```c++
virtual const word& name() const = 0;               // for the log line
virtual label solve(const coupledLinearSystem& sys, // returns linear iterations
                    scalarField& x) = 0;
static autoPtr<coupledSolver> New(const dictionary& dict);
```

`x` is sized `nLocalRows`, holds the initial guess on entry and the solution
on return. A direct solve returns 1.

`New` reads `linearSolver`, defaulting to `PETSc` when `Pstream::parRun()` and
`SparseLU` otherwise. It raises a fatal error for `PETSc` in a build without
`-DHAVE_PETSC`, and for an Eigen backend in a parallel run; anything else
falls through to the Eigen constructor, which rejects unknown names.

### 2.3 Adding a backend

1. Write `myCoupledSolver.H` / `.C` deriving from `Foam::coupledSolver`.
   Include the third-party headers **first** in the `.C`, before any OpenFOAM
   header, and keep the library's types out of the `.H` behind a `void*`
   handle - `eigenCoupledSolver.H` is the pattern. This avoids macro
   collisions and keeps the header cheap to include.
2. Read the backend's controls from the same `dictionary` the factory was
   given. Backend keys are siblings of the solver keys, not a nested
   sub-dictionary. Give every key a default and document it in
   `doc/USAGE.md`. Check for a name collision first: `tolerance` and
   `maxLinearIter` already mean different things with different defaults in
   the two existing backends.
3. Consume the system through the accessors. A per-entry backend uses
   `row(k)`, `col(k)`, `val(k)`; a batched backend uses `rowData()`,
   `colData()`, `valData()` and registers the pattern once.
4. If the dependency is optional, wrap the whole translation unit in
   `#ifdef HAVE_<DEP>` so it compiles to an empty object file without it, and
   have the factory emit a fatal error naming the missing dependency.
   `petscCoupledSolver.C` is the pattern.
5. Add the `.C` to `Make/files`, the discovery logic to `Make/options`
   (section 4), and a branch to `coupledSolver::New`.
6. Honour two contracts: the initial guess in `x` must be used (a Picard outer
   loop converges from it), and the sparsity pattern is fixed for the run, so
   any caching keyed on it is safe but must be validated - PETSc checks that
   the triplet *count* is unchanged and fails loudly; the Eigen direct path
   caches the symbolic factorisation and checks nothing.

### 2.4 Why the assembly is backend-agnostic

- **One assembly serves serial and MPI.** In serial, global indices are
  identical to local ones, so the abstraction costs nothing on the serial
  path. The evidence that it is genuinely the same code is that parallel runs
  reproduce serial outer-iteration counts exactly - not within one - and
  reproduce fields to within about 2e-10 (section 5.4).
- **The assembly never needs the sparsity pattern.** Appending triplets with
  summed duplicates means no insertion search and no CSR bookkeeping in the
  physics code. That is what allows a boundary condition to contribute to a
  diagonal that internal faces also touch, without either side knowing about
  the other.
- **It is also the GPU-friendly path.** `MatSetPreallocationCOO` +
  `MatSetValuesCOO` is PETSc's device assembly API: the pattern is registered
  once, and each iteration hands over one contiguous value array. Switching
  the linear algebra onto a device needs no change on the OpenFOAM side at
  all - only `-mat_type aijcusparse -vec_type cuda` in `petscOptions`.
  Profiling confirms the handoff is free: `MatSetValuesCOO` measures at 1 % of
  wall time with zero logged host-device transfers.
- **The value is API fit, not insertion speed.** Replacing per-entry
  `MatSetValues` calls with the batched COO path saved about 0.1 s per
  iteration (0.7 %) on a 483k-cell case. The insertion loop was never the
  bottleneck; roughly 70 % of per-iteration wall time is preconditioner setup
  and application. Optimising the assembly further is not where the time is.


## 3. OpenFOAM 14 API notes

Practical facts, all verified against the OpenFOAM 14 sources. Most of this
section exists because the information is not in the doxygen.

### 3.1 Extracting coefficients from an `fvMatrix`

An `fvVectorMatrix` is not a block matrix. It carries one scalar `lduMatrix`
shared by all three components, plus per-component boundary coefficients:

| Piece | Type | Destination in the coupled matrix |
| --- | --- | --- |
| `UEqn.diag()[celli]` | `scalarField` | `(4c+i, 4c+i)` for `i = 0,1,2` |
| `UEqn.upper()[facei]` | `scalarField` | `A[own][nei]` |
| `UEqn.lower()[facei]` | `scalarField` | `A[nei][own]` |
| `UEqn.source()[celli][c]` | `vectorField` | RHS |
| `UEqn.internalCoeffs()[patchi][i][c]` | `vectorField` | diagonal of the face cell, on **all** patches |
| `UEqn.boundaryCoeffs()[patchi][i][c]` | `vectorField` | RHS on uncoupled patches; **negated** into the remote column on coupled patches |

Addressing: `mesh.lduAddr().lowerAddr()` is the owner list and
`upperAddr()` the neighbour list, identical to `mesh.owner()` and
`mesh.neighbour()` for an `fvMesh`. `upper()[f]` is the coefficient in the
owner's row and the neighbour's column.

Traps:

- `lower()` returns the upper coefficients when the matrix is symmetric (a
  pure Laplacian). The present operator always contains `fvm::div`, so it is
  asymmetric and the two arrays are distinct - but code that assumes they
  always are will alias silently if the operator is ever made symmetric.
- `diag()` is shared by the three components while `internalCoeffs()` is not.
  That asymmetry is the whole reason `aP` has to be assembled by hand rather
  than taken from `fvMatrix::A()`.
- `A()` and `H()` are not used anywhere in this solver: both bake in a
  division by cell volume and a relaxation history that a coupled assembly
  must not inherit.

### 3.2 The coupled-patch sign convention

On a coupled patch (processor or cyclic), the momentum contribution is

```
diagonal            +=  internalCoeffs[patchi][i][c]
remote cell column  +=  -boundaryCoeffs[patchi][i][c]
```

The minus sign follows `processorFvPatchField::updateInterfaceMatrix`, which
performs `result[faceCells] -= coeffs*pnf`: in a segregated solve the coupled
contribution is *subtracted* from the matrix-vector product, so assembling the
same operator explicitly requires the negated coefficient in the column.

**This is the classic parallel trap**, for three reasons. The sign is not
derivable from the coefficient names. It is invisible in serial, because the
branch is never taken. And getting it wrong does not produce a slightly worse
answer - it produces a qualitatively different solution or a divergent solve.
The only reliable detection is a parallel-versus-serial comparison, which is
why that check is a mandatory gate (section 5.4) rather than an optional one.
Measured agreement is 1e-10 or better, so any sign or indexing error in this
branch is unmissable.

The remote cell's global index comes from
`syncTools::swapBoundaryCellList(mesh, gCell, nbrGCell)` and is indexed
`fvPatch::start() - mesh.nInternalFaces() + i`.

### 3.3 Boundary conditions: use the coefficient functions, never the type

Do not branch on the boundary-condition type. Every `fvPatchField` exposes the
splitting its own discretisation uses:

| Function | Meaning | Arguments |
| --- | --- | --- |
| `valueInternalCoeffs(w)` | implicit part of the face **value** | interpolation weights |
| `valueBoundaryCoeffs(w)` | explicit part of the face **value** | interpolation weights |
| `gradientInternalCoeffs()` | implicit part of the face-normal **gradient** | none (already carries `deltaCoeffs`) |
| `gradientBoundaryCoeffs()` | explicit part of the face-normal **gradient** | none |

Using these makes the assembly exact for every boundary-condition type that
`fvc::grad` and `fvm::laplacian` already handle, and makes cyclic and
processor support fall out for free. Concretely, a `fixedValue` pressure patch
contributes `+Df*|Sf|*deltaCoeffs` to the owner diagonal and the corresponding
term to the RHS, while a `zeroGradient` pressure patch contributes nothing -
without either case being named in the code.

Three practical notes:

- They return `tmp<Field>`. Bind the `tmp` to a named object and take a const
  reference from it; do not take a reference to the temporary directly.
- `fixesValue()` is the right predicate for behavioural decisions:
  `U.boundaryField()[patchi].fixesValue()` switches the Rhie-Chow pressure
  terms off where the flux is exactly known (inlets, no-slip walls), and
  `p.boundaryField()[patchi].fixesValue()` decides whether the pressure level
  needs pinning.
- Empty patches have `fvPatch::size() == 0`, so patch loops skip them with no
  special case. Empty *directions* are different: `mesh.solutionD()[c] == -1`
  needs an identity momentum row `U_c = 0`, matching what
  `fvMatrix::solveSegregated` does when it skips an invalid component. The
  visible consequence is a permanent `Solving for Uz, Initial residual = 0`
  line on a 2-D case.

**Where the generic path is not enough.** Impermeable constraint patches -
`symmetry`, `symmetryPlane`, `wedge`, `slip` - carry exactly zero mass flux,
but their coefficients are per-component and diagonal-only, which cannot
express zero *normal* flux on a face that is not axis-aligned. On an
unstructured mesh the result is a mass leak through the plane. Hence the
`zeroFluxPatch` predicate, which forces both the continuity contribution and
the reconstructed flux to zero on those patches, treating them like walls.
This is not cosmetic: before the predicate existed, a 483k-cell unstructured
case converged cleanly to a pressure drop 9.6 % low. `partialSlip` is not
covered by the predicate and would leak.

### 3.4 `globalIndex` and `syncTools`

- `globalIndex globalCells(mesh.nCells())` builds the rank offsets;
  `toGlobal(celli)` maps local to global and `size()` is the global cell
  count. In serial the mapping is the identity.
- `syncTools::swapBoundaryCellList(mesh, cellData, nbrCellData)` returns, for
  every boundary face, the value in the cell on the far side. It handles
  processor **and** cyclic patches, which is why the assembly needs no
  separate cyclic path. It is used three ways: once at start-up for the remote
  global cell indices, and twice per outer iteration for remote `U` and `p` in
  the flux reconstruction and the residual matrix-vector product.
- The 1:1 face-pairing assumption is implicit and unchecked. Any patch whose
  `fvPatch::coupled()` is true but whose face mapping is not one-to-one -
  `cyclicAMI` and other non-conformal couplings - takes the coupled branch and
  produces a wrong operator with no diagnostic. Adding a detection-and-fail
  guard is a well-scoped contribution (section 6.6).
- Reductions: `reduce(x, sumOp<scalar>())` for residual sums,
  `reduce(flag, maxOp<label>())` for the pressure-level flag. The latter is
  not optional - a patch that fixes `p` may exist on only some ranks.

### 3.5 Momentum transport and viscosity

Construct OpenFOAM's own framework rather than a hand-rolled viscosity:

```c++
autoPtr<viscosityModel> viscosity(viscosityModel::New(mesh));

autoPtr<incompressible::momentumTransportModel> momentumTransport
(
    incompressible::momentumTransportModel::New(U, phi, viscosity)
);

momentumTransport->validate();
```

Exact names that work, and the ones that do not:

- `viscosityModel.H` is in `$(LIB_SRC)/physicalProperties/lnInclude`.
  `viscosityModel::New(mesh)` reads the `viscosityModel` keyword.
- The class is `Foam::incompressibleMomentumTransportModel`, header
  `incompressibleMomentumTransportModel.H`. The short spelling
  `incompressible::momentumTransportModel` works because that header ends with
  a `typedef` inside `namespace incompressible`.
- **The third selector argument is `const Foam::viscosity&`, not
  `const viscosityModel&`.** Passing the `autoPtr<viscosityModel>` directly
  compiles - `autoPtr::operator const T&()` is the one user-defined
  conversion, and `viscosityModel` to `viscosity` is a standard derived-to-base
  conversion - and is exactly what the stock solver module does.
- **The method is `divDevSigma(U)`, not `divDevTau(U)`**, on the incompressible
  base. `divDevSigma` is a one-line alias for `divDevTau`, but it is what
  `incompressibleFluid/momentumPredictor.C` calls, and matching the stock
  module is the point.
- **`validate()` must be called after construction.** For `eddyViscosity` it
  is `correctNut()`, making `nut` consistent with the initial `k` and
  `epsilon` before the first matrix is assembled.
- **`predict()` is deliberately not called.** In OpenFOAM 14 the only
  non-trivial overrides are `generalisedNewtonian` and the
  `laminarModel`/`Stokes` chain, all no-ops for eddy-viscosity RAS models, and
  the base is an empty definition. `correct()` at the end of every outer
  iteration already leaves `nut` current. A model whose `predict()` did real
  work would not get it.
- **Case files are mandatory**, laminar cases included:
  `constant/momentumTransport` with `simulationType` (falling back to
  `constant/turbulenceProperties`) and `constant/physicalProperties` with
  `viscosityModel` (falling back to `thermophysicalProperties` then
  `transportProperties`, handled by `physicalProperties::findModelDict`). If
  neither momentum-transport file exists the run dies in the `IOdictionary`
  read.
- **Turbulence residuals for convergence control**: `Foam::convergenceControl`
  in `libfiniteVolume` exposes two **public static** functions that need only
  an `fvMesh`, with no `solutionControl` constructed -
  `getFieldNames(mesh)` and
  `getInitialResiduals(mesh, fieldName, solvei, r0, r)`. The underlying
  `getInitialTypeResiduals` silently leaves `r0` and `r` untouched for an
  unknown field, so guarding the lookup with `getFieldNames` is required; that
  is also exactly how the stock `criteriaSatisfied()` decides what counts as
  "checked".

**Operator equivalence for laminar.** Worth knowing before touching the
momentum operator, because it is what makes the laminar bit-level regression
gate meaningful. Unrolling the OpenFOAM 14 sources:

```
divDevSigma(U) = divDevTau(U) = DivDevTau(rho_ = 1, U)
               = divDevTauCorr(-alphaRhoNuEff*fvc::dotInterpolate(Sf, dev2(T(grad U))), U)
                 - fvm::laplacian(alphaRhoNuEff, U)
divDevTauCorr(X, U) = fvm::divc(surfaceVectorField(X), U)
fvm::divc(tflux, vf) = fvm::Su(fvc::div(tflux), vf)
fvm::Su(su, vf)      : source() -= mesh.V()*su, nothing else
alphaRhoNuEff        = fvc::interpolate(alpha_*rho*nuEff())
```

Three identities make this the same matrix as a hand-written
`fvm::div(phi,U) - fvm::laplacian(nuf,U)` plus an explicit `dev2(T(grad U))`
source, bit for bit:

1. `fvc::interpolate(uniform nu) == nuf` **bitwise** - OpenFOAM's weighted
   interpolation is coded as `lambda*(v_P - v_N) + v_N`, which returns `v`
   exactly for a uniform field, so no `w + (1-w) != 1` rounding appears.
2. `fvc::dotInterpolate(Sf, T) == Sf & fvc::interpolate(T)` - same weights,
   same scheme lookup, fused into one loop.
3. `fvm::Su(fvc::div(flux), U)` writes only into `source()` and contributes
   nothing to `diag`, `upper`, `lower`, `internalCoeffs` or `boundaryCoeffs`.

For a RAS model the same expression carries `nuEff = nu + nut` and brings the
wall functions with it. `div(devTauCorrFlux)` needs **no `divSchemes` entry**:
`fvc::div` of a surface field is `fvc::surfaceIntegrate`, not a Gauss scheme.
Legacy `div(devTauCorrFlux)` or `div((nu*dev2(T(grad(U)))))` entries in a
case's `fvSchemes` are vestigial and harmless.

### 3.6 OpenFOAM 14 changes that break ported code

| Symptom | Fact |
| --- | --- |
| `fvCFD.H: No such file` | OpenFOAM 14 does not ship it. Include `argList.H`, `Time.H`, `fvMesh.H`, `volFields.H`, `surfaceFields.H`, `fvm.H`, `fvc.H`, `fvMatrices.H` explicitly |
| `dimViscosity` undeclared | Removed from `dimensions.H`; use `dimArea/dimTime` |
| `fvPatch::patch()` not found | No such accessor; use `fvPatch::start()` for the boundary-face offset |
| `Time::constantPath()` not found | Removed. `physicalProperties::findModelDict` already probes the alternative dictionary names |
| `IOobject::typeHeaderOk` inaccessible | Protected; not callable from application code |
| `call of overloaded cbrt(double) is ambiguous` | `Foam::cbrt` clashes with `::cbrt`; use `std::cbrt` |
| `Make/options:N: *** missing separator` | `Make/options` is read raw by GNU make. C-style `/* */` comments break it; use `#` |
| No `simpleFoam` binary | Foundation OpenFOAM 14 has no standalone `simpleFoam`. The segregated reference is `foamRun -solver incompressibleFluid` with `SIMPLE` dictionaries |
| Tutorial mesh dictionaries missing | Shared `blockMeshDict` files live under `$FOAM_TUTORIALS/resources/blockMesh/` |

### 3.7 Backend library gotchas

- **Eigen**: `SparseLU::info()` must not be called after `analyzePattern()`.
  `m_isInitialized` is set only by `factorize()`, so the assertion fires.
- **PETSc**: assigning `PETSC_COMM_WORLD = MPI_COMM_WORLD` before `MPI_Init`
  fails with `PETSC_ERR_SUP`. In a serial OpenFOAM run MPI is not initialised,
  so the assignment must be guarded by `MPI_Initialized()`; PETSc initialises
  MPI itself in that case.
- **Some PETSc options are consumed inside `PetscInitialize`** and cannot come
  from the dictionary string, which is inserted afterwards.
  `-use_gpu_aware_mpi` is set by the constructor with `PetscOptionsSetValue`
  before `PetscInitialize` (explicitly legal); `-log_view` must come from the
  `PETSC_OPTIONS` environment variable.
- **`VecGetArrayWrite` / `VecRestoreArrayWrite`, not `VecGetArray`**, for the
  RHS and initial guess. Every local entry is overwritten, so the
  device-to-host fetch `VecGetArray` performs is pure waste on a `VECCUDA` -
  two full vector copies per outer iteration. Semantics are identical on a
  host `Vec`.
- **`MatSetFromOptions` must follow `MatSetType`** so that `-mat_type
  aijcusparse` can replace the default. The same ordering applies to
  `VecSetFromOptions`.
- **`MatSetBlockSize(A, 4)`** is what gives `PCFIELDSPLIT` the `[u,v,w]` and
  `[p]` index sets for free. Nothing about the split is coded; it is entirely
  option-driven.
- **PETSc 3.19 to 3.25 API skew: none.** `MatSetPreallocationCOO`,
  `MatSetValuesCOO`, `KSPSetTolerances`, `PETSC_DEFAULT` and the
  `VecGetArrayWrite` family are unchanged.
- **Precision contract**: a `static_assert` requires
  `sizeof(scalar) == sizeof(PetscScalar)`, because the COO value array is
  handed over by `reinterpret_cast` with no copy. A single-precision or
  complex PETSc fails to compile, which is the intent. `PetscInt` against
  `Foam::label` width is **not** asserted - only `static_cast`s - so a 32-bit
  PETSc combined with `WM_LABEL_SIZE=64` would truncate silently once the
  global row count exceeds 2^31, i.e. above roughly 5e8 cells.
- **`mpirun` drains stdin.** A script piped into a shell loses everything
  after the first `mpirun` unless each one gets `< /dev/null`.


## 4. Build system

### 4.1 `Make/files`

Four translation units - `coupledLinearSystem.C`, `eigenCoupledSolver.C`,
`petscCoupledSolver.C`, `coupledSimpleFoam.C` - and
`EXE = $(FOAM_USER_APPBIN)/coupledSimpleFoam`. The executable installs into
the user app bin, not the platform bin, so no OpenFOAM installation is
modified.

### 4.2 How `Make/options` selects backends

`Make/options` is a GNU makefile fragment, so it can compute. It does four
things:

1. **Eigen include.** `EIGEN_INC ?= -I/usr/include/eigen3`. Eigen is header
   only and always required; the default is the Debian/Ubuntu location and
   `EIGEN_INC` overrides it for any other layout.
2. **PETSc discovery.** By default, `pkg-config --cflags PETSc` with a
   fallback to lowercase `petsc` (Debian ships the capitalised `.pc` name).
   Nothing is hardcoded and nothing needs editing to enable PETSc: install it
   and rebuild.
3. **CUDA PETSc.** With `PETSC_CUDA=1`, the query instead runs against
   `$(PETSC_CUDA_DIR)/$(PETSC_CUDA_ARCH)/lib/pkgconfig` - defaulting to the
   standard `PETSC_DIR` / `PETSC_ARCH` variables - because that `.pc` file
   carries the full library list for hypre, cuSPARSE and anything else
   configured in. If pkg-config finds nothing it falls back to bare `-lpetsc`
   with explicit include and library paths. Either way it appends
   `-Wl,-rpath` to that library directory, since a source build is not on the
   loader path.
4. **Disable.** `COUPLED_NO_PETSC=1` blanks both flag sets and wins over
   `PETSC_CUDA`.

MPI flags come from `mpicc --showme:compile` and `--showme:link`, because
PETSc's `.pc` file does not carry the MPI include path. `--showme` is
OpenMPI-specific, which matches `WM_MPLIB=SYSTEMOPENMPI`; on an MPICH stack
these come back empty and the MPI include path has to be supplied by hand.

If PETSc flags end up non-empty, the build adds `-DHAVE_PETSC` and
`-DOMPI_SKIP_MPICXX` (the latter keeps the deprecated OpenMPI C++ bindings out
of the OpenFOAM translation units).

**`-DHAVE_PETSC` is the only thing that enables the PETSc backend.** Both
`petscCoupledSolver.H` and `.C` wrap their entire contents in it, so the
translation unit compiles to an empty object file when PETSc is absent, and
`coupledSolver::New` reports a runtime error naming the fix. There is no
partially-built state.

### 4.3 Environment switches

| Switch | Effect |
| --- | --- |
| *(none)* | PETSc auto-detected through pkg-config; CPU only |
| `COUPLED_NO_PETSC=1` | Eigen-only build; wins over everything else |
| `PETSC_CUDA=1` | Link a CUDA-enabled PETSc |
| `PETSC_DIR` / `PETSC_ARCH` | Standard PETSc variables naming that build |
| `PETSC_CUDA_DIR` / `PETSC_CUDA_ARCH` | Override the above for this build only, when two PETSc installations must coexist |
| `EIGEN_INC` | Eigen include flag |
| `PETSC_OPTIONS` | Runtime, not build time. The only route for `-log_view` |

All three build configurations produce the **same executable name** and
overwrite each other in `$FOAM_USER_APPBIN`. Always `./Allclean` when
switching, because the include paths differ, and confirm which one is
installed with `ldd $FOAM_USER_APPBIN/coupledSimpleFoam | grep -iE "petsc|cuda"`.

Note the asymmetry that follows: CPU versus GPU **linear algebra** is a
runtime choice (`-mat_type aijcusparse -vec_type cuda` in `petscOptions`), but
CPU versus CUDA **PETSc** is a compile-time choice.

### 4.4 Library set

```
-lmomentumTransportModels                 base classes and every wall-function fvPatchField
-lincompressibleMomentumTransportModels   laminar / RAS / LES instantiations
-lphysicalProperties                      viscosityModel
-lfiniteVolume
-lmeshTools                               syncTools, wedge and symmetry patches
```

This is exactly what the stock `incompressibleFluid` solver module links.
Those five are sufficient for every incompressible model: `kEpsilon`,
`kOmegaSST`, `realizableKE`, `SpalartAllmaras` and their `nut*` / `epsilon*` /
`omega*` wall functions all register themselves through the run-time selection
tables.

### 4.5 Adding a dependency

1. Discover it in `Make/options` into a local variable, preferably through
   `pkg-config`, and leave that variable empty when it is not found. Never
   bake an absolute path into the repository; provide an environment-variable
   override instead, as `EIGEN_INC` does.
2. Append the variable to `EXE_INC` and `EXE_LIBS`. An absent dependency then
   contributes nothing and the build still succeeds.
3. Define a `-DHAVE_<DEP>` macro only when the flags are non-empty, and guard
   every line of the dependent code with it, so the translation unit compiles
   to nothing and the failure surfaces as a clear runtime message rather than
   a link error.
4. Add the switch to the configuration echo in `Allwmake` and document it in
   `doc/INSTALL.md` and in the `Make/options` header comment.

Remember that `Make/options` is read raw by GNU make: `#` comments only.


## 5. Testing and regression protocol

These gates are mandatory. They exist because the failure modes of this solver
are quiet: a wrong sign in a parallel branch, a mass leak through a constraint
patch, and a broken matrix-to-flux pairing all produce a run that converges
and prints plausible numbers.

Run the whole tutorial suite with `./Allrun` from the repository root. It runs
every `tutorials/<case>/Allrun`, writes `<case>/log.Allrun`, prints a pass/fail
summary and exits non-zero if any case failed, so it is directly usable as a
regression gate. A case fails on a non-zero exit **or** on `FOAM FATAL ERROR`
appearing in any of its logs. Individual cases run with
`cd tutorials/<case> && ./Allrun`.

Field comparisons use `tools/compare_fields.py`:

```sh
tools/compare_fields.py <caseA> <timeA> <caseB> <timeB> [n]
```

It reports the largest cell-wise velocity difference anywhere in the domain
and the `Ux` profile along the vertical centreline of a structured 2-D mesh.

### 5.1 Gate 0: it builds in every configuration

```sh
./Allclean && ./Allwmake                        # default: PETSc if present
./Allclean && COUPLED_NO_PETSC=1 ./Allwmake     # Eigen only
```

Add `PETSC_CUDA=1` if a CUDA PETSc is available. A change that only compiles
with PETSc present has broken the `#ifdef HAVE_PETSC` isolation.

### 5.2 Gate 1: laminar bit-level regression

**This is the strongest gate in the repository and the cheapest to run.** The
Eigen `SparseLU` backend is a deterministic direct solve, so a serial laminar
run is reproducible bit for bit. If the assembled matrix is unchanged, the log
is byte-identical.

```sh
cd tutorials/cavity && ./Allrun && cp log.coupledSimpleFoam /tmp/before.log
# apply the change, rebuild
./Allrun && diff /tmp/before.log log.coupledSimpleFoam     # must be empty
```

Compare `Solving for`, `time step continuity errors` and
`coupled assembly check` lines. Run it on **two topologies**: the lid-driven
cavity (`tutorials/cavity`), which is a single wall-type patch on a structured
mesh, and the interrupted-fin channel (`tutorials/stripChannel`), which
exercises inlet, outlet, wall and internal fin patches. The second is the
decisive one - a bit-identical log over 25 outer iterations on a case with
that patch mix proves the assembled matrix is unchanged.

Reference outer-iteration counts, serial Eigen `SparseLU`, at
`residualControl U 1e-5, p 1e-4`:

| Case | Cells | Coupled outer iterations |
| --- | --- | --- |
| Lid-driven cavity 64x64 | 4,096 | 12 |
| Lid-driven cavity 128x128 | 16,384 | 12 |
| Lid-driven cavity 256x256 | 65,536 | 12 |
| Backward-facing step, laminar | 12,225 | 22 |
| stripChannel, Re 400 | 10,920 | 19 |
| stripChannel, Re 1000 | 10,920 | 25 |

The stripChannel Re 1000 inlet pressure drop is `7.92849560e-01`. That value
is reproduced by the serial Eigen direct solve, by serial PETSc, and by PETSc
at 4 and 6 ranks, to every printed digit; it is the single most useful
reference number in the suite because any change to the operator moves it.

A changed iteration count is not automatically a failure, but it must be
explained numerically, and a changed converged answer in serial beyond
roundoff is a failure unless the change is a deliberate, documented
discretisation change.

### 5.3 Gate 2: mass conservation

Every outer iteration prints

```
coupled assembly check: sum|div(phi)| = ... (excl. pRef cell ...), max|div(phi)| = ..., normalised = ...
```

The requirement is machine precision, normalised by the characteristic flux.
Reference, lid-driven cavity 64x64, first iteration:

```
time step continuity errors : sum local = 1.28080978574e-16, global = -8.98498470606e-19, ...
coupled assembly check: sum|div(phi)| = 1.28080978574e-16 (excl. pRef cell 1.2643671557e-16), max|div(phi)| = 1.64426300419e-18, normalised = 1.75162884398e-16
```

It stays at that level on every subsequent iteration, and it stays there in
RANS with a strongly varying `nuEff` (2.2e-17 at iteration 1 and 1.4e-16 at
iteration 779 on the RANS backward-facing step). The original acceptance
threshold was `1e-10 * characteristic flux`, so there is nine orders of
headroom; anything that moves this to 1e-9 has broken something structural.

The usual cause is the flux reconstruction no longer using exactly the same
`Df`, `Seff` and lagged gradient as the matrix. The second usual cause is a
new boundary-condition path that contributes to the continuity row but not to
the reconstruction, or the reverse.

Two legitimate exceptions, both tolerance effects rather than defects: the
"excl. pRef cell" and full sums agree to roundoff, because the remaining N-1
continuity rows plus a zero net boundary flux force the pinned cell's
divergence to zero too; and a loose inner tolerance degrades the figure
directly - RANS at `relTol 1e-6` gives `sum|div(phi)| = 1.8e-9` (normalised
2.8e-8) where `relTol 1e-12` gives ~3e-15.

### 5.4 Gate 3: parallel versus serial consistency

Mandatory for **any** change that touches the assembly, the boundary loops or
the indexing, because it is the only check that exercises the coupled-patch
branch (section 3.2).

```sh
decomposePar                                   # scotch or simple
mpirun -np 4 coupledSimpleFoam -parallel < /dev/null
reconstructPar -latestTime
tools/compare_fields.py <serialCase> <t> <parallelCase> <t> <n>
```

Two requirements:

- **Outer-iteration counts must be identical to serial**, not within one.
- **Field agreement at 1e-8 or better.**

Measured, laminar cavities against the serial Eigen direct solve:

| Comparison | Centreline max abs(dUx) | All-cell max abs(dU) |
| --- | --- | --- |
| Cavity 64x64, np4 | 9.00e-12 | 1.21e-11 |
| Cavity 128x128, np4 | 3.70e-11 | 5.71e-11 |
| Cavity 128x128, np6 | 3.40e-11 | 5.01e-11 |
| Cavity 256x256, np4 | 2.05e-10 | 2.09e-10 |
| Cavity 256x256, np6 | 2.03e-10 | 2.07e-10 |
| Cavity 64x64, serial PETSc vs serial Eigen | 9.00e-12 | 9.55e-12 |

The floor is partly the ASCII `writePrecision 12` in `controlDict`, i.e. about
1e-12 on values of order one. A sign or index error in the coupled-patch
branch does not degrade these figures - it changes the solution.

At the tolerances used in RANS the counts are not bit-identical: the RANS
backward-facing step converges in 779 outer iterations serially and 771 at
np4, with inlet pressures agreeing to 0.006 %. That is expected, because
PETSc's reductions are not run-to-run reproducible; a 1 % spread in the final
iteration count is acceptable there, a different answer is not.

### 5.5 Gate 4: agreement with the segregated solver

Required for changes to the discretisation. Run `foamRun -solver
incompressibleFluid` on the same case with the same `residualControl` - the
tutorial cases carry both sets of dictionaries, since `foamRun` ignores the
extra `COUPLED` sub-dictionary.

Two things must be understood before reading the result, or the gate produces
false alarms:

- **The reference's own convergence level dominates.** Segregated residuals
  decay at roughly 0.993 per iteration near convergence, so stopping at
  `U 1e-5` still leaves about 1e-3 of solution error. Judge against a tight
  reference. On the cavity, the coupled-versus-segregated centreline
  difference is 8.82e-4 against a reference stopped at `U 1e-5 / p 1e-4` and
  **1.03e-5** against the same case run to `U 1e-9 / p 1e-9`.
- **The Rhie-Chow damping definitions differ by design.** Segregated SIMPLE
  takes `rAU = 1/UEqn.A()` from the *relaxed* momentum matrix, so its
  effective damping is `alpha_U * V/aP`; a coupled solve has no relaxation
  factor and uses `V/aP`. The two differ by an O(h^2) Rhie-Chow term that is
  largest where `rAU` jumps hardest between neighbours. Setting
  `rhieChowFactor` to the reference run's momentum relaxation factor
  reproduces the segregated discretisation and isolates real differences from
  this one: on the cavity it collapses the centreline difference from 8.82e-4
  to 1.43e-6 and the all-cell difference from 1.40e-2 (a single singular
  lid-corner cell) to 9.60e-4.

With that understood, expected pressure-drop agreement at production criteria
is 0.17 % to 0.38 % across the laminar 2-D suite, and 0.11 % on the 3-D
interrupted-fin unit cell once the damping definition is matched. The
mechanism is confirmed independently in three settings, including RANS, where
the gap falls from 0.678 % to 0.075 % at matched damping - larger than in
laminar cases because `nuEff` spans orders of magnitude and `rAU` therefore
jumps harder.

### 5.6 Gate 5: RANS smoke

Required when a change touches the momentum operator or the turbulence path.
On a `kEpsilon` case, check all of:

- `Selecting momentum transport model type RAS` and
  `Selecting RAS turbulence model kEpsilon` appear;
- `k` and `epsilon` solve every outer iteration with monotonically decreasing
  initial residuals;
- zero `bounding k` and `bounding epsilon` warnings, no NaN, no `sigFpe`;
- `sum|div(phi)|` still at machine precision (gate 2);
- the log keeps the standard OpenFOAM residual format, so `foamLog` and
  PyFoam parse it unchanged.

Reference: the RANS backward-facing step (12,225 cells, `kEpsilon`) converges
in 779 outer iterations serially against 1,997 for segregated SIMPLE at
identical `residualControl`.

Two expected behaviours that are not defects. Convergence has two phases - a
fast one where the coupled solver reaches `Ux < 1.7e-03` in 61 outer
iterations against SIMPLE's 451, then a long plateau at `Ux ~ 5e-05` that
collapses only once `k` and `epsilon` reach about 1e-05. The plateau is the
lagged turbulence, not the linear solve: freezing `nut` and restarting makes
the coupled p-U block converge geometrically at about 0.82 per iteration,
reaching 3e-08 in 40 iterations. Section 6.3 gives the mechanism. And on small
2-D RANS cases the segregated solver wins on wall clock despite needing 2.6x
more iterations, which is a documented property, not a regression.

### 5.7 Gate 6: performance claims

If the claim is performance, the protocol is:

- **Report ratios, not absolute times**, and time both binaries back to back
  under identical machine load, best of three. Absolute wall-clock is not
  comparable across measurements: a shared host inflates times by 1.5x to
  3.5x while leaving iteration counts and pressures untouched.
- **State mesh, cell count, backend, rank count, outer iterations and seconds
  per outer iteration.** `doc/BENCHMARKS.md` sets the format.

For calibration, changing the momentum operator to `divDevSigma` measured as a
ratio of 1.055 on the cavity and 1.008 on stripChannel - a 1-5 % cost from the
extra `fvVectorMatrix` temporaries - and that is the kind of resolution the
back-to-back protocol gives.


## 6. Extension roadmap

Concrete directions, with what the existing measurements actually say about
each. All performance figures below are for the reference cases in
`doc/BENCHMARKS.md`.

### 6.1 A custom block-AMG preconditioner with block smoothers

**Difficulty: high.** This is a research-scale project - a coupled algebraic
multigrid over the block-4 system with block-ILU or Vanka-type smoothers, in
the tradition of block-selective AMG for implicitly coupled pressure-velocity
systems and of solvers such as ICSFoam that build the hierarchy themselves.
PETSc's generic components do not provide it.

What the evidence says:

- **It is where the remaining time is.** Profiling puts roughly 70 % of
  per-iteration wall time in the preconditioner - about 44 % in one AMG solve
  on the Schur block and 25 % in velocity-block ILU setup. Assembly is 1 %.
- **Generic scalar AMG on the whole block system does not work.** GAMG applied
  monolithically had to be killed after 20 minutes for a single iteration.
  Scalar smoothed aggregation with point smoothers cannot handle the
  stabilised saddle-point structure.
- **The one AMG that does work on the hard topology is nodal systems-AMG**,
  which reproduces the reference answer on the interrupted-fin channel but at
  per-inner cost comparable to ASM, and it has no device implementation.
- **The design constraint is cheap setup, not reuse.** Freezing the
  preconditioner across outer iterations costs +53 % wall time, and freezing
  only the hierarchy structure costs +66 %. The Picard operator changes fast
  enough that a stale preconditioner buys more inner iterations than it saves
  in setup. A block-AMG must therefore make *fresh* setup intrinsically cheap
  - pairwise or double-pairwise aggregation is the architecture reference -
  rather than amortise setup across iterations.
- **The bar has moved.** The shipped fieldsplit-Schur with hypre BoomerAMG on
  the Schur block already delivered 2.7x over the previous default from a
  one-line options change. A custom block-AMG competes against that, not
  against the old baseline.

### 6.2 Adaptive inner tolerances

**Difficulty: low to moderate**, and the highest ratio of value to effort on
this list. `KSPSetTolerances` is already called on every solve, so the change
is a policy plus a control loop, contained in `petscCoupledSolver::solve` and
its dictionary keys.

What the evidence says:

- **A single constant cannot work.** The measured recipe is different in four
  regimes: `relTol 1e-12` for small laminar cases; `1e-6` for large laminar
  cases with pseudo-transient continuation; `1e-12` for small RANS; and
  `1e-8` with `maxLinearIter 500` for large 3-D RANS, where the tight setting
  needs about 2,340 inner iterations per outer solve and is intractable.
- **Getting it wrong fails in both directions.** Too loose stalls the outer
  Picard loop outright - `1e-3` and `1e-6` both failed to converge on the
  cavity, because a loose linear update makes the outer correction negligible.
  Too tight makes each outer iteration unaffordable.
- **The payoff is measured.** On the 483k-cell case, moving from `1e-12` to
  `1e-6` alongside pseudo-transient continuation took the run from 496 s to
  259 s at the same outer-iteration count - 1.9x.
- **The schedule must be coupled to the damping.** A control run with a loose
  inner tolerance and no pseudo-transient damping plateaued at about
  6e-6 / 1e-4 after 80 outer iterations. Damping is what converts loose inner
  solves into convergence, so an adaptive scheme cannot ignore it.
- Eisenstat-Walker forcing terms are the standard formulation for exactly this
  problem and are unexplored here.

### 6.3 Coupling turbulence into the block system

**Difficulty: high**, and structurally invasive: the block size changes (4 to
6 for k-epsilon), which touches the index arithmetic, the fieldsplit index
sets, the residual machinery, the COO pattern and both backends.

What the evidence says:

- **The lag genuinely sets the convergence floor.** The coupled residual is
  `b_{n+1} - A_{n+1} x_n`, and `A_{n+1}` contains `Df = V/aP` built from
  `nuEff = nu + nut`. While `nut` changes, the operator changes between
  iterations and the residual cannot vanish however well the p-U block is
  solved. Freezing `nut` and restarting collapses the plateau immediately
  (section 5.6): 4e-05 to 3e-08 in 40 iterations, at about 0.82 per iteration.
- **The same floor appears in the segregated solver**, so it is a property of
  lagged RANS rather than of this solver.
- **But the payoff is not obvious.** Per-iteration cost already dominates: on
  a 12k-cell 2-D RANS case the coupled solver needs 779 outer iterations
  against 1,997 for SIMPLE and still loses on wall clock, 389 s against 61 s.
  A 6x6 block system raises per-iteration cost further, and the preconditioner
  question (section 6.1) would have to be answered for the larger block before
  the iteration saving could be realised.
- The honest scope is: documented as future work, with a clear mechanism and
  an unproven payoff.

### 6.4 Compressible and transient extension

**Transient: moderate.** The machinery is largely present - pseudo-transient
continuation is already a per-cell `V/dtau` diagonal with the matching
`V/dtau*U_old` RHS term, so a genuine `fvm::ddt` is a small change to that
code path plus removing the `ddtSchemes steadyState` requirement. What is not
present is any of the validation: every reference value, every preconditioner
recommendation and every convergence recipe in this repository is for steady
flow and would have to be re-established.

**Compressible: high.** The pressure equation changes character, density
enters the coupling, and both the block structure and the `selfp` Schur
approximation would need rethinking. The existing preconditioner evidence does
not transfer.

### 6.5 GPU work

**Difficulty: the path already works.** On a CUDA-enabled PETSc build, adding
`-mat_type aijcusparse -vec_type cuda` to `petscOptions` moves the entire
linear-algebra path onto the device with no rebuild of the solver logic,
because the COO assembly is already PETSc's device assembly API.

What the evidence says - and this is scoped to one workstation-class card with
1/64-rate FP64, under WSL2, with no MPS and a non-CUDA-aware MPI, so it should
not be extrapolated:

- **It is numerically exact.** At `relTol 1e-12` the device path reproduces
  the serial direct-solve reference to all printed digits.
- **It is not faster.** At fixed preconditioner and rank count the GPU is 7 %
  slower (8.25 against 7.72 s per outer iteration); over full runs it ties
  (94.3 s at np4 on the device against 94.7 s at np6 on the host); on a
  43,680-unknown problem it is about 30x slower, far below the kernel-launch
  break-even. It wins only where the host preconditioner is weak, which is a
  win over a worse algorithm.
- **Scaling is inverted with rank count** - 3.55, 4.28 then 5.90 s per outer
  iteration at 4, 6 and 8 ranks - because without MPS each rank opens its own
  CUDA context on one device and they time-slice. The host control over the
  same range is flat to improving.
- **The structural reason is data movement in the preconditioner.** About
  1.53 GB is copied device-to-host per run, because `PCFIELDSPLIT` extracts
  submatrices and hypre needs its own format, both round-tripping every outer
  iteration. Since the matrix is rebuilt and re-shipped every Picard step,
  there is nothing to amortise. Utilisation is 97-99 %, so this is not an
  occupancy problem.
- **Both routes to a device-resident *block* AMG are closed on that
  platform.** A GPU-native aggregation AMG segfaults inside the virtualised
  CUDA driver during matrix upload, and its aggregation selectors resolve to
  host-only implementations; hypre's nodal systems-AMG - the only AMG that
  reproduces the reference on the hard 2-D topology - has no device
  implementation. What remains on the device is scalar AMG on the Schur
  complement, which is what ships.
- **What would change the answer**: bare-metal Linux with a working
  block-AMG library; MPS, or one rank per device; a part with 1/2-rate rather
  than 1/64-rate FP64; and a mesh large enough that the per-Picard matrix
  re-ship stops dominating. None of those applied to the measurements above.

The honest summary: the 2.7x that was actually achieved came from the
algorithm - swapping GAMG for hypre BoomerAMG on the Schur block - and was
entirely on the CPU.

### 6.6 Smaller, well-scoped items

Good first contributions. Each is a documented rough edge with a known fix.

- **Detect unsupported coupled patches.** `cyclicAMI` and other non-conformal
  couplings take the `fvPatch::coupled()` branch, which assumes 1:1 face
  pairing, and produce a wrong operator silently. At minimum this should be a
  fatal error naming the patch.
- **`partialSlip` in `zeroFluxPatch`.** It is impermeable but not in the
  predicate, so it leaks mass exactly as symmetry patches did before the
  predicate existed.
- **Bounds-check `pRefCell`.** It is a *global* index, which differs from the
  usual OpenFOAM convention. Out of range, `ownRow()` is false on every rank,
  `setRowIdentity` becomes a no-op and the matrix stays singular, surfacing as
  a confusing factorisation failure.
- **Wedge transforms in the pressure-gradient block.** Wedge patches go down
  the generic `valueInternalCoeffs` path, which does not account for their
  component-mixing transform. They are at least zero-flux-correct in the
  continuity row.
- **A consistency check on the flux balance.** There is no `adjustPhi`
  equivalent, so a fixed-flux domain with an inconsistent inflow/outflow
  balance produces a singular continuity block with no diagnostic.
- **Non-orthogonal correctors.** The non-orthogonal Rhie-Chow correction is
  fully explicit and lagged one outer iteration, with no
  `nNonOrthogonalCorrectors` equivalent. On strongly non-orthogonal meshes
  this is the accuracy and robustness ceiling.
- **`fvModels` / `fvConstraints` support.** Currently not assembled at all,
  and not warned about, so porous zones, momentum sources and MRF are silently
  ignored if a case provides them.

Several of these interact with the fixed COO sparsity pattern, so check
section 2.1 before starting.


## 7. Coding conventions

### 7.1 OpenFOAM style

The solver is read alongside OpenFOAM's own sources, so it follows the
Foundation house style:

- 4-space indent, no tabs; 80-column limit; LF line endings; **ASCII only** -
  no typographic dashes or quotes, and no byte-order mark (a UTF-8 BOM ahead
  of the banner comment breaks some tooling).
- Opening brace on its own line. One blank line between functions, two between
  sections. The `// * * *` separator comments delimit constructor, destructor
  and member-function sections.
- Every file carries the standard header block with the GPL-3.0 notice, and
  headers carry `Class` / `Description` / `SourceFiles` blocks. An application
  `.C` carries `Application` / `Description`, and for `coupledSimpleFoam.C`
  that `Description` block is the specification of the discretisation - keep
  it in step with the code.
- `Foam::` types throughout: `label`, `scalar`, `word`, `List`, `Field`,
  `DynamicList`. Use `//-` for member documentation comments.
- Third-party headers are included **first** in a `.C`, never in a `.H`, and
  their types are kept out of headers behind opaque handles.

### 7.2 Comment discipline

**Comments explain constraints and conventions, not code.** A reader can see
what a line does; what they cannot see is what would go wrong if it were
written the obvious way. Four categories earn a comment:

1. **Conventions imported from elsewhere** that cannot be derived from the
   code. The negated `boundaryCoeffs` on coupled patches is the canonical
   example: the comment names
   `processorFvPatchField::updateInterfaceMatrix` and its
   `result[faceCells] -= coeffs*pnf`, so the sign can be verified against a
   source rather than trusted.
2. **Invariants another part of the code depends on.** That `aP` is used only
   for `Df` while the matrix receives `diag()` and `internalCoeffs`
   separately; that the flux reconstruction must use the same `Df` and the
   same lagged gradient as the assembly; that the COO pattern is identical on
   every outer iteration.
3. **Deliberate omissions that look like bugs.** `predict()` is not called;
   `A()` and `H()` are not used; the converged iteration prints
   `Final residual` equal to `Initial residual`; the assembly check line is a
   permanent verification print. Without a comment, each of these invites a
   well-meaning "fix".
4. **Measured justifications for a default.** The `petscOptions` default
   carries the topologies it was validated on and what the alternatives do
   instead, because the next person to change it needs to know which cases it
   has to keep passing.

What does not earn a comment: restating the line (`// loop over cells`),
narrating history (dates, who changed what, "fixed the leak"), or attribution.
State the constraint that holds now, in present tense. If a comment quotes a
number, it must be a measurement that can be reproduced from
`doc/BENCHMARKS.md` or from the tutorials.

### 7.3 Two rules specific to this solver

- **The assembly is backend-agnostic.** Nothing in `coupledSimpleFoam.C` may
  mention Eigen or PETSc, by name or by type. New backends implement
  `Foam::coupledSolver` and are selected by the `linearSolver` keyword in the
  factory.
- **The COO sparsity pattern is identical on every outer iteration.** Both
  backends cache work keyed on it. Anything that would add or drop a triplet
  mid-run - topology change, adaptive refinement, a boundary condition that
  switches between fixed value and zero gradient - needs that caching reworked
  first, in both backends.

### 7.4 Naming and output

Follow the symbols already in use, and keep the C++ and the reference
prototype in step so the two can be read side by side: `aP` for the momentum
diagonal, `Df` for the interpolated Rhie-Chow coefficient, `rAU` for `V/aP`,
`Seff` for the effective area vector of the lagged term, `gradPf` for the
interpolated lagged gradient.

The `Info<<` residual lines deliberately mimic the segregated solver's format
so that `foamLog` and PyFoam parse the log unchanged. Do not "tidy" them. New
diagnostic output should either follow an existing OpenFOAM line format or be
clearly labelled, as the `coupled assembly check` line is.
