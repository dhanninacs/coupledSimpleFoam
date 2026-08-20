# heatedDuct — plane channel with isothermal walls, Re = 20, Pr = 7

The heat-transfer case, and the one tutorial with an **exact answer to check
against**. Fully developed laminar flow between two parallel plates held at a
constant and equal temperature has a Nusselt number of 7.5407; this case
reproduces it. If `solveEnergy` is ever broken, this is where it shows.

## What it demonstrates

- **`solveEnergy`**, the temperature transport equation solved segregated
  after the coupled p-U solve, on the conservative Rhie-Chow flux that solve
  produced. Three keys in `COUPLED` and a `0/T` file are the whole setup.
- **The passive-scalar structure.** Constant properties and no buoyancy, so
  `T` never enters the momentum equations. Setting `solveEnergy no` in this
  same case changes the velocity and pressure by nothing at all.
- **The `coupled energy check` log line**, the temperature counterpart of
  `coupled assembly check`: it reports the wall heat flux and closes the
  boundary energy balance every outer iteration.
- **An exact physics gate**, Nu = 7.5407, rather than a code-to-code
  comparison.

## Physics and boundary conditions

A plane channel 10 m long and H = 1 m high, 2-D, laminar, incompressible. The
hydraulic diameter of an infinite parallel-plate duct is Dh = 2H = 2 m, so the
domain is 5 hydraulic diameters long.

| Quantity | Value |
|---|---|
| Inlet velocity | 1 m/s, uniform |
| ν | 0.1 m²/s |
| Re = U·Dh/ν | 20 |
| Pr (`COUPLED/Pr`) | 7 — water-like |
| Pe = Re·Pr | 140 |
| α = ν/Pr | 0.0142857 m²/s |
| Inlet temperature | 300 K |
| Wall temperature | 350 K |

| Patch | Type | U | p | T |
|---|---|---|---|---|
| `inlet` | patch | `fixedValue (1 0 0)` | `zeroGradient` | `fixedValue 300` |
| `outlet` | patch | `zeroGradient` | `fixedValue 0` | `zeroGradient` |
| `walls` | wall | `noSlip` | `zeroGradient` | `fixedValue 350` |
| `frontAndBack` | empty | `empty` | `empty` | `empty` |

The mesh is a uniform 200 × 40 × 1 block: **8 000 cells**, giving a coupled
p-U system of **32 000 unknowns** and a segregated temperature system of 8 000.

### Why these numbers, and not rounder ones

The three conditions the Nu = 7.5407 benchmark needs pull against each other,
and the case is sized so that all three hold at once in a domain small enough
to run in seconds.

- **The flow must be hydrodynamically developed.** The entry length is about
  0.05·Re·Dh = 2 m, so the velocity profile is parabolic well before the
  stations where Nu is evaluated. This is why Re is low.
- **Axial conduction must be negligible.** The benchmark assumes Pe → ∞. The
  ratio of axial conduction to axial convection for the developed field is
  about 7.54/Pe_H² with Pe_H = u·H/α = 70, i.e. 0.15 %. This is why Pr is 7
  rather than 0.71: at Re = 20 and Pr = 0.71 the Peclet number would be 14 and
  axial conduction would shift Nu by roughly 15 %, towards the low-Peclet
  limit of 8.117.
- **The temperature profile must be developed, and `Tw - Tb` must still be
  resolvable.** The local Nu is within a per cent of its asymptote by
  x/(Dh·Pe) ≈ 0.02, i.e. x ≈ 5.6 m, and `Tw - Tb` decays as
  exp(−7.54·α·x/(u·H²)), so at the outlet it is still about a quarter of its
  inlet value. There is no thermal-saturation cancellation in the reported
  numbers.

A uniform mesh in y is enough for an accurate wall heat flux here, and this is
worth knowing: for the fully developed profile the energy equation is
u ∂T/∂x = α ∂²T/∂y², and u = 0 at a no-slip wall, so **the temperature profile
has exactly zero curvature at the wall**. The leading error term of the
one-sided wall gradient vanishes with it, making the estimate second-order
accurate rather than first.

## How to run it

```sh
./Allrun      # blockMesh, copy 0.orig to 0, coupledSimpleFoam, report
./Allclean    # remove every generated file
```

For the segregated comparison, from this same directory, adding a
`scalarTransport` function object to `controlDict` so that the reference
solves the same temperature equation:

```sh
foamRun       # controlDict already carries `solver incompressibleFluid`
```

`doc/VALIDATION.md` section 7 gives the exact function-object settings that
make the reference equation identical term for term, and what the comparison
measures.

## Expected outcome

Verified with OpenFOAM 14 (build `14-7b05503f98a8`), serial Eigen SparseLU
backend:

| Quantity | Value | Tolerance |
|---|---|---|
| Coupled outer iterations | **11** | exact; a different count means the setup changed |
| Coupled p-U system size | 32 000 unknowns | exact |
| Inlet-to-outlet Δp | **13.0918221595** m²/s² | to the printed digits |
| Net wall heat flux | **1.7944553015** K m³/s | to the printed digits |
| Energy-balance imbalance | 2.4e-09 % | order of magnitude; it is a linear-solver tolerance, not a physical quantity |
| Nu at x = 8.975 m | **7.5416** | against the exact 7.5407, i.e. +0.012 % |
| Final `sum\|div(phi)\|` | 4.9e-14 (normalised 4.8e-15) | order of magnitude |
| Segregated SIMPLE reference | **177** iterations | exact for the shipped `residualControl` |

The wall heat flux is **kinematic**, in K m³/s: it is
`alphaEff·grad(T)` integrated over the wall patches, that is a heat flux
divided by ρ·cp. Multiply by ρ·cp for watts.

### The Nusselt number along the duct

`Allrun` prints the local Nusselt number at four stations, computed by
`tools/heated_duct_nusselt.py` from the written `T` and `U` fields and the cell
centres:

```
         x               Tb          Tw - Tb     |dT/dy|_wall         Nu Nu/7.5407
    1.9750     313.81081883      36.18918117       143.928496     7.9542   1.05484
    3.9750     320.95244414      29.04755586       110.203319     7.5878   1.00624
    5.9750     326.58879122      23.41120878        88.337677     7.5466   1.00078
    8.9750     333.04854164      16.95145836        63.920434     7.5416   1.00012

Nu at x = 8.9750 : 7.5416   (exact fully developed value 7.5407, +0.01 %)
```

The post-processor itself is worth **+0.030 %** on this mesh: the exact
fully developed profile (the first Graetz eigenmode, integrated to 1e-12)
sampled at these same 200 × 40 cell centres and passed through
`heated_duct_nusselt.py` returns 7.5430 rather than 7.5407, because of the
one-sided wall gradient and the midpoint bulk integral. **7.543 is therefore
the number a correct solution produces here**, and the remaining gap to 7.5407
is post-processing rather than physics. `doc/VALIDATION.md` section 7.2 has the
calibration.

### The energy balance

Every outer iteration prints

```
coupled energy check: wall heat flux = 1.7944553015, enthalpy outflow = 1.73999043168, boundary diffusion = 1.73999043164, imbalance = 2.36837395898e-09 %
```

`enthalpy outflow` is `sum(phi_b·T_b)` over every boundary face — outlet minus
inlet — and `boundary diffusion` is the diffusive influx over every boundary,
which here is the wall heat flux plus the conduction through the `fixedValue`
inlet. Summing the assembled equation over all cells telescopes the internal
faces away and leaves exactly those two sums, so they must agree on a converged
solution; the reported imbalance is the level the `T` linear solve was asked
for, not a physical residual.

## Changing the wall condition

`0/T` documents the two alternatives inline. Measured on this same case:

| Wall condition | What happens | Measured |
|---|---|---|
| `fixedValue 350` (shipped) | isothermal wall | Nu → 7.5416 |
| `zeroGradient` | adiabatic wall | `T` stays at **exactly** 300 K in every cell (`max abs(T - 300) = 0`) and the wall heat flux is exactly 0 |
| `fixedGradient 175` | prescribed heat flux | wall heat flux exactly 2.5 K m³/s (`2·α·g·A_wall`), Nu = 8.2466 at x = 9.025 m against the exact 8.235 |

The constant-heat-flux case has its own exact answer, Nu = 8.235 for parallel
plates with equal flux on both walls, and reproduces it as well — see
`doc/VALIDATION.md` section 7.

## Runtime

Seconds. The mesh is 8 000 cells and the solve is 11 outer iterations of a direct sparse factorisation, so it finishes in about a second on an idle machine; the segregated reference needs 177 SIMPLE iterations. Both figures inflate several-fold on a loaded machine, so
treat them as an order of magnitude rather than a benchmark.

## Files worth reading

- `system/fvSolution` — the `COUPLED` dictionary, with `solveEnergy`, `Pr` and
  the `T` entry in `residualControl`; and the `solvers { T { ... } }` block,
  which **both** solvers read, because the temperature equation is segregated
  in either of them.
- `system/fvSchemes` — `div(phi,T)` is second-order upwind rather than the
  first-order upwind used for momentum, and the entry says why: at this cell
  size first-order upwind adds a streamwise numerical diffusivity of about
  twice the molecular one, which is enough to move Nu by a few tenths of a per
  cent.
- `0/T` — the thermal boundary conditions, and the two alternatives commented
  out in the `walls` entry.
