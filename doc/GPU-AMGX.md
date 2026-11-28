# GPU solving with NVIDIA AmgX

coupledSimpleFoam runs its whole linear-algebra path on one NVIDIA GPU through
[AmgX](https://github.com/NVIDIA/AMGX) (BSD-3-Clause), on Linux and on native
Windows (FoamWin), for plain flow, heat transfer and conjugate heat transfer.
This is a different route from the PETSc device path of `coupledSimpleFoam-gpu`
and replaces it as the GPU option: on the hardware measured there the PETSc
route only matched ten CPU cores, and on the public square duct below its
device preconditioner did not converge at all.

## What runs on the GPU

| Piece | Where | How |
| --- | --- | --- |
| The coupled p-U system | `linearSolver AmgX` | one 4x4 block-CSR matrix, BiCGStab + aggregation block AMG with multicolour block DILU smoothing. No field split, so none of PETSc's host-side submatrix extraction |
| A scalar equation (T, k, omega, ...) | `solver AmgX;` in `fvSolution` | an OpenFOAM `lduMatrix` solver: any equation of any solver linked with it. OpenFOAM's own residuals, measured on the host, so `residualControl` means what it always did |
| Conjugate heat transfer | `COUPLED/solidRegions` | the fluid and every solid region as **one** temperature system (`conjugateEnergy`), the interfaces written into the matrix, solved by AmgX (or Eigen on the CPU) |

The assembly stays on the host and is parallelised with OpenMP where it
dominated: the block scatter of the 4x4 values and the residual products run
over rows of a triplet grouping built once (the pattern is static).

## Using it

```c++
COUPLED
{
    linearSolver     AmgX;      // the coupled p-U system on the GPU
    amgxSmoother     GS;        // AMG smoother: GS (default) or DILU
    // amgxPrecision mixed is refused: AmgX 2.5's mixed mode fails here
    pcReuseInterval  8;         // rebuild the AMG hierarchy every 8 solves
    amgxReuse        resetup;   // refresh its values in between (default)
    // amgxConfig    "...";     // a full AmgX config; outer scope "main"

    solveEnergy      yes;
    Pr               0.7;
    // conjugate heat transfer: run with -region fluid
    solidRegions     (solid);   // regions from splitMeshRegions -cellZones
    rhoCp            1e5;       // fluid rho*Cp, J/m^3/K
    conjugateSolver  AmgX;      // or Eigen (CPU)
    conjugateTolerance 1e-9;
    timing           yes;       // per-section wall times in the log
}
```

The temperature of a non-conjugate case goes to the GPU with
`T { solver AmgX; tolerance 1e-12; relTol 0; maxIter 1000; }`.

A conjugate case is the multi-region layout `splitMeshRegions` writes: the
fluid region in coupledSimpleFoam's own format (kinematic `p`, constant `nu`),
solid regions with `kappa` in `constant/<solid>/physicalProperties`
(`mixture/transport/kappa`), the interface patches `<fluid>_to_<solid>` /
`<solid>_to_<fluid>` conformal and `fixedValue` (the solve sets them to the
interface temperature). Run `coupledSimpleFoam -region fluid`.

## AmgX for any OpenFOAM solver (transient runs, mesh changes)

`src/amgxSolvers` builds the scalar AmgX solver as a library, `libamgxSolvers`
(`./Allwmake` with `AMGX_DIR` set; FoamWin ships it in `bin/amgx`). Any solver can
load it, for example `foamRun -solver incompressibleFluid` in a transient
(PIMPLE) run:

```c++
// system/controlDict
libs ("libamgxSolvers.so");      // ".so" also on Windows: FoamWin loads the .dll

// system/fvSolution
p      { solver AmgX; tolerance 1e-6; relTol 0.05; }
pFinal { $p; relTol 0; }
pcorr  { solver AmgX; tolerance 1e-2; relTol 0; }   // with a topology changer
```

- The solver of each field is kept between solves: the AMG hierarchy is
  refreshed on the new coefficients (`resetup`) and rebuilt in full every
  `amgxSetupInterval` solves (default 10).
- A topology change (`fvMeshTopoChangers` refiner, mesh-to-mesh mapping)
  rebuilds the field's pattern and GPU handles. A 3D lid-driven cavity
  refined and unrefined five times solved its pressure and `pcorr` on AmgX
  throughout, through the same mesh changes as the GAMG run.
- Serial only, and no cyclic or processor interfaces (stopped with an error).
- **Use it on large meshes.** Every GPU solve carries a fixed latency of many
  small kernel launches and synchronisations. On the Re 100 cylinder tutorial
  (5,388 cells), AmgX pressure gives the same drag as PCG (Cd to 1e-4 before
  shedding starts) but runs about 9x slower than PCG on one core. On a mesh that
  small the GPU has nothing to hide its latency behind; it pays off at hundreds of
  thousands of cells and up, as for the coupled solver.

## Smoother, configuration and robustness

- `amgxSmoother GS` (the default since 2026-11-28) is multicolour Gauss-Seidel, relaxation 0.9,
  two pre- and post-sweeps. `amgxSmoother DILU` is the earlier multicolour DILU with one sweep.
  GS converged a stretched (aspect ratio ~10) micro-channel conjugate case on which DILU ran every
  linear solve to `maxLinearIter` and diverged (FoamWin #4). On the isotropic cases it costs 5-10 %:
  the 2.1M-cell duct took 52.3 s against 49.7 s, the 80k-cell conjugate duct 5.4 s against 5.0 s,
  with the same iteration counts. The same key sets the scalar (`solver AmgX;`) and conjugate solves
  (`conjugateSmoother` overrides it for the conjugate one).
- `amgxConfig "..."` replaces the whole built-in configuration (an AmgX config string, outer scope
  `main`), for the coupled system (in `COUPLED`) or a scalar equation (in its `solvers` entry):

  ```c++
  amgxConfig "config_version=2, exception_handling=1, main:convergence=RELATIVE_INI_CORE, main:norm=L2, main:monitor_residual=1, main:print_solve_stats=0, solver(main)=PBICGSTAB, main:preconditioner(amg)=AMG, amg:algorithm=AGGREGATION, amg:selector=SIZE_2, amg:smoother(sm)=MULTICOLOR_GS, sm:relaxation_factor=0.9, amg:presweeps=2, amg:postsweeps=2, amg:max_iters=1, amg:cycle=V, amg:max_levels=50, amg:coarse_solver=DENSE_LU_SOLVER, amg:dense_lu_num_rows=64, amg:min_coarse_rows=32";
  ```

- A non-finite solution from any AmgX solve stops the run at once with what to try, instead of
  failing later inside AmgX ("solver_resetup failed: Internal error").
- `amgxPrecision mixed` is refused: with AmgX 2.5 the mixed (dDFI) mode failed inside AmgX on every
  case tried (a floating-point exception in its convergence check for the block system, and
  "Configuration feature is not implemented" for scalar systems).
- Outer iterations: laminar ducts converge in 12-15, but cases with stretched cells, thin channels and
  tight `residualControl` (`T 1e-6`) have needed 50-80; the integral quantities (temperatures,
  pressure drop) stop changing well before the residual target. Monitor them, or loosen `T`.
- Mid-size meshes: at ~150k cells the host side (operators, assembly, and the conjugate energy system)
  costs more than the GPU solve, so a conjugate case there may run no faster than `foamMultiRun` on
  4 ranks. The GPU pays off from about a million cells.

## Measured (2026-11-27)

One NVIDIA RTX 4000 Ada (20 GB, 1/64-rate FP64) against an Intel i9-13900
(24 cores, dual-channel DDR5-5600), wall clock, each run alone. Public cases
only, generated by `tools/gpu-bench/`: a square duct 1 x 1 x 8 at Re 20, 64 x
64 x 512 = 2.1 M cells (isothermal, and heated with walls at 350 K, Pr 0.7),
and a conjugate duct (the same duct inside a 0.25-thick solid shell, k_s = 10
k_f, outer surface 350 K), 0.88 M fluid + 1.11 M solid cells.

Native Windows (FoamWin; CPU runs on 10 MS-MPI ranks, the GPU on one rank):

| Case | GPU (AmgX) | Best CPU solver | x | Stokes' default engine | x |
| --- | ---: | ---: | ---: | ---: | ---: |
| Duct, isothermal | **45.8 s** | 166.9 s (coupled, PETSc) | 3.6 | 904.3 s (`foamRun incompressibleFluid`, 810 its) | 19.7 |
| Duct, heated | **73.3 s** | 245.8 s (coupled, PETSc) | 3.4 | 1710.9 s (`foamRun fluid`, 1136 its) | 23.3 |
| Conjugate duct | **33.7 s** | - | - | 474.8 s (`foamMultiRun`, 713 its) | 14.1 |

Linux (WSL2), 10 OpenMPI ranks for the CPU:

| Case | GPU (AmgX) | Best CPU solver | x | Stokes' default engine | x |
| --- | ---: | ---: | ---: | ---: | ---: |
| Duct, isothermal | **55.8 s** | 179 s (coupled) | 3.2 | 998 s (`foamRun incompressibleFluid`, 826 its) | 17.9 |
| Duct, heated | **81.9 s** | 219 s (coupled) | 2.7 | - | - |
| Conjugate duct | **36.5 s** | - | - | 440 s (`foamMultiRun`, 714 its) | 12.1 |

"Best CPU solver" is coupledSimpleFoam itself with its tuned PETSc configuration (its
doc/BENCHMARKS.md 9.2); the conjugate case has no coupled CPU counterpart.

The answers are the CPU's: the heated duct's wall heat flux 76.70728 on every
coupled path, and its outlet temperature 349.9661 K against 349.9586 K from `foamRun fluid`; the isothermal pressure drop 13.167555 against 13.167557 (coupled CPU) and
13.2002 (segregated, its own discretisation); the conjugate interface
temperature 349.318155 K against foamMultiRun's 349.318089 K and the outlet
temperature within 0.005 K, on the same mesh. foamMultiRun's reference uses the
compressible `fluid` module at rho 1, Cp 1e5: with a small Cp its pressure-work
and kinetic-energy terms are worth kelvins and it is no longer a passive scalar.

Per outer iteration on the 2.1 M isothermal duct (Linux) about 4.5 s: the GPU
solve ~0.8 s, AMG resetup ~0.3 s, upload ~0.2 s, and the rest host-side
OpenFOAM assembly (operators ~0.7 s, triplets ~0.5 s) and residuals. The
conjugate case spends ~1.6 s per iteration assembling its energy system on the
host. Those are the next places to look.

## Building

Linux: build AmgX (`cmake -DCMAKE_CUDA_ARCHITECTURES=<sm> -DCMAKE_NO_MPI=True`,
target `amgxsh`, install), then `AMGX_DIR=<prefix> wmake` here.

FoamWin: build `amgxsh.dll` natively with MSVC (Visual Studio Build Tools) and
the CUDA toolkit (same CMake options, Visual Studio generator), then
`AMGX_WINDOWS_BUILD=<folder with amgxsh.dll and amgxsh.lib> tools/build-foamwin-amgx.sh`
in the FoamWin cross-build environment. It installs
`bin/amgx/coupledSimpleFoam.exe` beside `amgxsh.dll`. AmgX's interface is plain
C, which is what lets the MinGW-built solver call the MSVC-built library. At
run time the CUDA toolkit's `bin` (cuBLAS, cuSPARSE, cuSOLVER) must be on PATH,
or those redistributable DLLs placed beside the executable.

## Limits

- One rank, one GPU. A parallel run uses PETSc.
- After a run, `cyclic flux check` lines report, per cyclic pair, the net flux and the largest
  |phi + phi_nbr|; it must be round-off, or mass (and heat) leaks across the pair.
- Conjugate interfaces must be conformal (every face matched by centre).
- Cyclic patches inside a region are coupled in the conjugate temperature system, as in the
  p-U system (conformal cyclic only; cyclicAMI and other coupled types stop the run with an
  error). Conjugate mode runs serially.
- The conjugate energy is a passive scalar: constant properties, no buoyancy,
  no radiation, no volumetric sources yet.
- `amgxReuse none` (keep the hierarchy untouched between rebuilds) does not
  converge: AmgX keeps its own reordered copy of the operator, so the
  coefficients must be refreshed with a resetup.
