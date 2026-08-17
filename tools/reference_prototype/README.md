# Reference prototype

`coupled_vs_simple.py` is an independent two-dimensional finite-volume
implementation of the same algorithm as the C++ solver, written from the
literature rather than from the solver sources. It exists to validate the
block assembly in `src/coupledSimpleFoam`: two implementations that were
written separately and agree are far stronger evidence than one implementation
that runs without crashing.

It is standalone. It does not import, link or read anything from OpenFOAM, and
nothing in the solver depends on it.

## What it does

A collocated 2-D channel with three interrupted fins blocked out of a
structured mesh, solved twice on the same mesh, with the same discretisation
(first-order upwind convection, central diffusion) and the same Rhie-Chow face
flux:

- `run_simple()` - segregated SIMPLE with velocity and pressure
  under-relaxation
- `run_coupled()` - momentum and continuity assembled as one block system and
  solved directly, optionally with pseudo-transient continuation

Because only the coupling strategy differs, the outer-iteration counts are
directly comparable, which is the point of the exercise.

`run_coupled()` is the executable specification of the C++ assembly. The three
structural choices it demonstrates are exactly those implemented in
`coupledSimpleFoam.C`:

1. an implicit Green-Gauss pressure-gradient block in the momentum rows
   (`Gx`, `Gy`),
2. a continuity row built from the Rhie-Chow flux and implicit in both
   velocity and pressure (`Du`, `Dv`, `App`),
3. the non-orthogonal / smoothing part of the Rhie-Chow term carried
   explicitly on the right-hand side, lagged one outer iteration.

The variable naming follows the C++ code, so the two can be read side by side.

## Running it

Requires Python 3 with NumPy and SciPy; no OpenFOAM environment is needed.

```sh
python3 coupled_vs_simple.py                 # 240 x 48 mesh, Re = 400
python3 coupled_vs_simple.py --re 1000       # higher Reynolds number
python3 coupled_vs_simple.py --quick         # 120 x 24, for a fast check
```

Iteration counts, wall-clock times and the inlet-to-outlet pressure drop for
each configuration are written to `results_re<Re>_<nx>x<ny>.json`.

## References

The discretisation and the coupled assembly follow the published treatments
cited in the module docstring: Moukalled, Mangani and Darwish for collocated
SIMPLE and Rhie-Chow, and Darwish, Sraj and Moukalled for the coupled
finite-volume assembly.
