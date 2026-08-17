# Contributing

coupledSimpleFoam is a single OpenFOAM application plus its tutorials and
documentation. The repository is small on purpose: one solver, one build
script, one tutorial runner.

## Building

Prerequisites:

- OpenFOAM 14 (openfoam.org), with its environment sourced
- Eigen 3 headers - the serial backend, always required
  (Debian/Ubuntu: `libeigen3-dev`)
- PETSc - optional, enables the MPI backend
  (Debian/Ubuntu: `petsc-dev`; the default preconditioner needs a PETSc
  configured with hypre)

```sh
. /opt/openfoam14/etc/bashrc     # path of the local OpenFOAM installation
./Allwmake
```

The executable is installed as `$FOAM_USER_APPBIN/coupledSimpleFoam`.

Build switches are environment variables:

| Variable | Effect |
| --- | --- |
| `COUPLED_NO_PETSC=1` | Serial Eigen-only build, even when PETSc is installed |
| `PETSC_CUDA=1` | Link a CUDA-enabled PETSc named by `PETSC_DIR` / `PETSC_ARCH` |
| `EIGEN_INC` | Override the Eigen include flag, e.g. `-I/opt/eigen/include` |

`./Allclean` removes the object tree and all tutorial run output.

Building inside a mounted Windows filesystem under WSL is roughly an order of
magnitude slower than building in the Linux filesystem; keep the working tree
in the Linux home directory.

## Running the tutorials

```sh
./Allrun
```

This runs every `tutorials/<case>/Allrun` (or `tutorials/<group>/<case>/Allrun`),
writes each case's console output to `<case>/log.Allrun`, and prints a
pass/fail summary. It exits non-zero if any case fails, so it is usable as a
regression gate. A case counts as failed when its `Allrun` exits non-zero or
when `FOAM FATAL ERROR` appears in any of its logs.

Individual cases can be run directly:

```sh
cd tutorials/<case>
./Allrun
```

`tools/compare_fields.py` compares the velocity fields of two case/time
directories cell by cell on a structured two-dimensional mesh, which is how
the coupled solution is checked against a segregated reference, a parallel run
against a serial one, or one linear-algebra backend against another.

## What a pull request should include

- **A clear statement of what changes and why.** Numerical changes are the
  interesting case: say which term of the discretisation moves, and whether
  the change is expected to alter converged results or only the iteration
  path.
- **Evidence that the tutorials still pass.** Paste the `./Allrun` summary. If
  a tutorial's converged result changes, say by how much and why that is
  correct.
- **A tutorial or an extension of one** for any new capability. A feature with
  no case that exercises it cannot be regression tested.
- **Documentation for anything user-visible.** New or changed `COUPLED`
  dictionary keys belong in `doc/USAGE.md`, and every key must have a
  documented default. New build switches belong in `doc/INSTALL.md`.
- **Performance numbers** when the claim is performance: mesh size, cell
  count, backend, number of ranks, outer iterations and wall-clock time per
  outer iteration. `doc/BENCHMARKS.md` sets the format.

## Code style

The solver follows the OpenFOAM Foundation house style, because it is read
alongside OpenFOAM's own sources:

- 4-space indent, no tabs, 80-column limit, LF line endings, ASCII only
- opening brace on its own line; one blank line between functions, two between
  sections
- every file carries the standard header block with the GPL-3.0 notice
- `Foam::` types throughout: `label`, `scalar`, `word`, `List`, `Field`
- comments explain why a term is where it is, not what the line does

Two conventions are specific to this solver:

- **The assembly is backend-agnostic.** Nothing in `coupledSimpleFoam.C` may
  mention Eigen or PETSc. New backends implement `Foam::coupledSolver` and are
  selected by the `linearSolver` keyword in the factory.
- **The COO sparsity pattern must be identical on every outer iteration.**
  Both backends cache work that depends on it: PETSc registers the pattern
  once with `MatSetPreallocationCOO`, and the Eigen direct solve analyses it
  once. Anything that would add or drop a triplet mid-run - topology change,
  a boundary condition that switches between fixed value and zero gradient -
  needs the caching reworked first.

## Scope

The solver is deliberately narrow: steady, incompressible, single phase, no
energy equation, and turbulence handled segregated through OpenFOAM's own
`momentumTransport` models. `fvModels`, `fvConstraints` and MRF are not
assembled. `doc/USAGE.md` lists the limitations in full; changes that widen
the scope are welcome but should start as an issue, since several of them
interact with the fixed sparsity pattern above.
