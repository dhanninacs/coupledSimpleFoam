# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses
[semantic versioning](https://semver.org/).

## 0.9.0 - initial public release

### Added

- Fully coupled steady incompressible p-U solver for OpenFOAM 14 (Foundation):
  block-4 `[u, v, w, p]` system with an implicit pressure-gradient block and a
  Rhie-Chow continuity row, assembled as backend-agnostic COO triplets.
- Serial Eigen backend (`SparseLU`, `BiCGSTAB`) and MPI PETSc backend
  (`MATMPIAIJ` + `KSP`), selected automatically or through `linearSolver`.
- Optional CUDA-enabled PETSc path.
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
