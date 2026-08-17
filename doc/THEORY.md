# Theory and algorithm

This document is the mathematical reference for `coupledSimpleFoam`: a
steady-state incompressible solver for OpenFOAM 14 that assembles the momentum
and continuity equations into **one** implicitly coupled linear system of block
size 4, with the unknowns interleaved per cell as $[u, v, w, p]$.

It assumes familiarity with the cell-centred finite volume method and with
SIMPLE. It describes what the solver does and why, term by term, at the level of
detail needed to reproduce the discretisation or to reason about a convergence
failure. Practical matters — dictionary keys, build switches, measured
performance — live in [`USAGE.md`](USAGE.md), [`INSTALL.md`](INSTALL.md) and
[`BENCHMARKS.md`](BENCHMARKS.md). Every work cited by author and year appears in
[`REFERENCES.md`](REFERENCES.md).

---

## Contents

1. [Problem statement](#1-problem-statement)
2. [Segregated versus coupled](#2-segregated-versus-coupled)
3. [Collocated grids and Rhie-Chow interpolation](#3-collocated-grids-and-rhie-chow-interpolation)
4. [The coupled system](#4-the-coupled-system)
5. [Boundary conditions](#5-boundary-conditions)
6. [Solution algorithm](#6-solution-algorithm)
7. [Turbulence](#7-turbulence)
8. [Linear algebra](#8-linear-algebra)
9. [Relationship to other implementations](#9-relationship-to-other-implementations)

---

## Nomenclature

Symbols are used consistently throughout. Vectors are bold; discrete
(matrix/vector) objects are upright capitals.

| Symbol | Meaning |
|---|---|
| $\mathbf{u}$, $u_c$ | velocity and its Cartesian component $c \in \{0,1,2\}$ ($u,v,w$) |
| $p$ | kinematic pressure, $\tilde{p}/\rho$ (dimensions $\mathrm{m^2\,s^{-2}}$) |
| $\nu$, $\nu_t$, $\nu_{\mathrm{eff}}$ | molecular, turbulent and effective kinematic viscosity, $\nu_{\mathrm{eff}} = \nu + \nu_t$ |
| $P$, $N$ | owner and neighbour cell of a face |
| $V_P$ | volume of cell $P$ |
| $\partial P$ | the set of faces of cell $P$ |
| $f$ | a face; $\mathbf{S}_f$ its outward area vector (owner-to-neighbour), $\lvert\mathbf{S}_f\rvert$ its area, $\hat{\mathbf{n}}_f = \mathbf{S}_f/\lvert\mathbf{S}_f\rvert$ |
| $s_{f,P}$ | face orientation sign: $+1$ if $P$ owns $f$, $-1$ otherwise |
| $\mathbf{d}_f$ | $\mathbf{C}_N - \mathbf{C}_P$, the owner-to-neighbour centre vector (on a patch, `fvPatch::delta()`) |
| $w_f$ | interpolation weight: $\varphi_f = w_f\varphi_P + (1-w_f)\varphi_N$ (`mesh.weights()`) |
| $\Delta_f$ | non-orthogonal delta coefficient (`mesh.nonOrthDeltaCoeffs()`), $=1/\lvert\mathbf{d}_f\rvert$ on an orthogonal mesh |
| $\mathbf{S}^{\mathrm{eff}}_f$ | $\lvert\mathbf{S}_f\rvert\,\Delta_f\,\mathbf{d}_f$; equals $\mathbf{S}_f$ on an orthogonal mesh |
| $\phi_f$ | volumetric face flux, $\mathrm{m^3\,s^{-1}}$ |
| $a_P$ | momentum diagonal coefficient of cell $P$ |
| $D_f$ | Rhie-Chow coefficient, the face interpolate of $\gamma V_P/a_P$ |
| $C_f$ | $D_f\lvert\mathbf{S}_f\rvert\Delta_f$, the implicit pressure-Laplacian face coefficient |
| $\gamma$ | `rhieChowFactor` (default $1$) |
| $\alpha_U$, $\alpha_p$ | segregated under-relaxation factors for momentum and pressure |
| $\Delta\tau_P$ | local pseudo-time step of cell $P$ |
| $\mathrm{F}$, $\mathrm{G}$, $\mathrm{D}$, $\mathrm{C}$ | momentum, gradient, divergence and pressure blocks of the coupled matrix |
| $\mathrm{S}$ | Schur complement of the coupled matrix |
| $\mathrm{K}$, $\mathbf{x}$, $\mathbf{b}$ | coupled matrix, unknown vector and right-hand side |
| $n$ | outer (Picard) iteration index |
| $h$ | representative cell size, $V_P^{1/3}$ |

---

## 1. Problem statement

### 1.1 The continuous equations

The solver targets the steady, incompressible, single-phase Navier-Stokes
equations in the kinematic (density-divided) form OpenFOAM uses:

$$
\nabla\cdot(\mathbf{u}\otimes\mathbf{u})
\;-\;\nabla\cdot\left(\nu_{\mathrm{eff}}\nabla\mathbf{u}\right)
\;-\;\nabla\cdot\left(\nu_{\mathrm{eff}}\,\mathrm{dev}_2\!\left[(\nabla\mathbf{u})^{\mathsf{T}}\right]\right)
\;+\;\nabla p \;=\; \mathbf{0}
$$

$$
\nabla\cdot\mathbf{u} \;=\; 0
$$

with $\mathrm{dev}_2(\mathrm{T}) = \mathrm{T} - \tfrac{2}{3}\,\mathrm{tr}(\mathrm{T})\,\mathrm{I}$.
The third term of the momentum equation is the transpose part of the deviatoric
stress; it vanishes identically for constant $\nu_{\mathrm{eff}}$ and a
divergence-free field, but is retained because the solver uses OpenFOAM's own
`divDevSigma` operator (§7.2), which carries it for every viscosity model.

Because the flow is incompressible and the density constant, $\rho$ never
appears: $p$ is a *kinematic* pressure and the momentum equation is a transport
equation for velocity alone.

### 1.2 Why pressure-velocity coupling is hard

There are four unknown fields ($u$, $v$, $w$, $p$) and four equations, so the
system is closed. The difficulty is structural, not one of counting:

* The momentum equations are transport equations **for velocity**. Pressure
  enters them only as a source, through $\nabla p$.
* The continuity equation contains **no pressure at all**. It is a constraint on
  the velocity field, not an evolution equation for anything.

There is therefore no equation that can be regarded as "the pressure equation",
and in particular nothing that puts $p$ on the diagonal of its own row. The
elliptic pressure equation that every incompressible algorithm eventually solves
is not one of the governing equations: it must be *manufactured* by taking the
divergence of the momentum equation and substituting continuity, which is an
algebraic consequence of the system rather than a member of it.

### 1.3 The discrete consequence: saddle-point structure

Discretise on a mesh of $N$ cells and order the unknowns as
$\mathbf{x} = (\mathbf{u}, \mathbf{p})$. The momentum rows are linear in
$\mathbf{u}$ (after linearising the convective term, §6.1) and linear in
$\mathbf{p}$; the continuity rows are linear in $\mathbf{u}$ and, in the exact
discretisation, contain no $\mathbf{p}$ at all. The system is

$$
\begin{pmatrix} \mathrm{F} & \mathrm{G} \\ \mathrm{D} & 0 \end{pmatrix}
\begin{pmatrix} \mathbf{u} \\ \mathbf{p} \end{pmatrix}
=
\begin{pmatrix} \mathbf{f} \\ \mathbf{0} \end{pmatrix}
$$

where $\mathrm{F}$ is the discrete momentum operator, $\mathrm{G}$ the discrete
gradient and $\mathrm{D}$ the discrete divergence. This is a **saddle-point
system** (Benzi, Golub & Liesen 2005). Its properties dominate everything that
follows:

* **The $(2,2)$ block is zero.** The matrix has no diagonal in its pressure rows
  and is *indefinite*: its eigenvalues have both signs. Every solver technique
  that assumes an M-matrix, diagonal dominance, or a definite operator — Jacobi
  and Gauss-Seidel smoothing, incomplete factorisation without pivoting,
  classical or smoothed-aggregation algebraic multigrid — is out of its
  hypotheses here (§8).
* **The system is well posed only if $\mathrm{D}$ and $\mathrm{G}$ satisfy a
  discrete inf-sup (LBB) condition.** On a collocated arrangement the naive
  choices do not, which produces the checkerboard mode (§3.1).
* **Pressure is determined only up to a constant** unless a Dirichlet pressure
  boundary condition is present, because $\mathrm{G}$ annihilates the constant
  vector on a closed domain (§5.6).

What this solver actually assembles is the *stabilised* saddle-point form

$$
\mathrm{K} =
\begin{pmatrix} \mathrm{F} & \mathrm{G} \\ \mathrm{D} & \mathrm{C} \end{pmatrix},
\qquad
\mathrm{C} \succ 0 \;\; \text{small},
$$

where $\mathrm{C}$ is the compact pressure Laplacian contributed by the
Rhie-Chow interpolation (§3). Negating the continuity rows puts it in the
canonical form $\bigl(\begin{smallmatrix}\mathrm{F} & \mathrm{G}\\ -\mathrm{D} &
-\mathrm{C}\end{smallmatrix}\bigr)$, whose symmetric part is the block-diagonal
matrix $\mathrm{diag}\bigl(\tfrac{1}{2}(\mathrm{F}+\mathrm{F}^{\mathsf{T}}),\,
-\mathrm{C}\bigr)$ — positive on the velocity block, negative on the pressure
block. The indefiniteness is intrinsic and cannot be relaxed away; it is a
property of the incompressibility constraint.

---

## 2. Segregated versus coupled

### 2.1 What SIMPLE does

SIMPLE (Patankar 1980) breaks the saddle-point system by *not* solving it.
Writing the momentum row of cell $P$ as

$$
a_P\,\mathbf{u}_P + \sum_{N} a_N\,\mathbf{u}_N \;=\; \mathbf{s}_P - V_P(\nabla p)_P
\qquad\Longleftrightarrow\qquad
\mathbf{u}_P \;=\; \frac{\mathbf{H}(\mathbf{u})_P}{a_P} - \frac{V_P}{a_P}(\nabla p)_P
$$

with $\mathbf{H}(\mathbf{u})_P = \mathbf{s}_P - \sum_N a_N \mathbf{u}_N$, SIMPLE:

1. solves the momentum equations with the **previous** pressure, obtaining
   $\mathbf{u}^*$, which does not satisfy continuity;
2. substitutes the above relation into the discrete continuity equation while
   **dropping the dependence of $\mathbf{H}$ on the velocity correction**,
   yielding a pressure(-correction) equation whose operator is
   $\mathrm{D}\,\widehat{\mathrm{F}}^{-1}\mathrm{G}$ with
   $\widehat{\mathrm{F}} = \mathrm{diag}(\mathrm{F})$;
3. corrects velocity and flux, under-relaxes, and repeats.

In linear-algebra terms SIMPLE is a **stationary iteration driven by a
block-triangular preconditioner** built on an approximate Schur complement. The
approximation — replacing $\mathrm{F}$ by its diagonal — is exactly what makes
the method cheap and exactly what makes it need under-relaxation: the neglected
terms feed back as an amplification that must be damped, typically with
$\alpha_U \approx 0.7$ on momentum and $\alpha_p \approx 0.3$ on pressure. The
price is paid in outer iterations, and the number of outer iterations is not
mesh-independent: refinement makes the dropped coupling relatively stronger. A
two-dimensional reference implementation of the identical discretisation
measures SIMPLE going from 149 to 284 iterations under one uniform refinement
while the coupled loop stays at 27-31 across the same refinement and
Reynolds-number range.

### 2.2 What a coupled solve does instead

The coupled solver assembles $\mathrm{F}$, $\mathrm{G}$, $\mathrm{D}$ and
$\mathrm{C}$ into a single matrix and hands it to a linear solver. There is no
Schur approximation in the *algorithm*, no pressure-correction step, and no
under-relaxation factor for $p$ or $\mathbf{U}$ — the momentum matrix is
assembled unrelaxed. Approximation reappears only inside the **preconditioner**
(§8), where it is harmless: an inexact preconditioner slows the Krylov solver
but cannot change the fixed point.

The remaining outer loop exists solely because the system is nonlinear (the
convective flux $\phi$) and because two terms are deliberately lagged (§4.4). It
is a Picard iteration, not a pressure-velocity decoupling.

### 2.3 The same discrete solution, a different path

This point is worth stating precisely because it is often muddled.

Both algorithms drive the residual of the **same set of discrete nonlinear
equations** to zero. Any fixed point of SIMPLE is a solution of those equations,
and so is any fixed point of the coupled loop. Coupling changes the *path* — the
iteration matrix, the contraction rate, the need for relaxation — and not the
*answer*. Under-relaxation factors are properties of the path only: at
convergence $\mathbf{u}^{n+1} = \mathbf{u}^{n}$ and every relaxation term
cancels identically.

**Except for one term, and it is instructive.** OpenFOAM's segregated solver
forms $r_{A_U} = 1/\mathrm{A}()$ from the *relaxed* momentum matrix, whose
diagonal is $a_P/\alpha_U$. The Rhie-Chow coefficient that survives to
convergence is therefore

$$
D_f^{\text{segregated}} \;=\; \left(\frac{\alpha_U V_P}{a_P}\right)_f ,
\qquad\text{not}\qquad
D_f^{\text{coupled}} \;=\; \left(\frac{V_P}{a_P}\right)_f .
$$

The relaxation factor does *not* cancel here, because $r_{A_U}$ is used to build
a flux, not to damp an update. The Rhie-Chow term is an $O(h^2)$ perturbation of
the discrete continuity equation (§3.3), so the two converged solutions differ by
an $O(h^2)$ term: a discretisation difference, not an error in either solver.
The fingerprint is documented in Moukalled, Mangani & Darwish (2016, §15.9.1).

The `rhieChowFactor` key exists to reproduce it. Setting $\gamma = \alpha_U$
makes $D_f$ identical to the reference run's and collapses the difference. This
has been quantified in three independent settings:

| case | difference at $\gamma = 1$ | difference at $\gamma = \alpha_U$ |
|---|---|---|
| lid-driven cavity, all-cell $\max\lvert\Delta u\rvert$ | $1.40\times10^{-2}$ (lid corner) | $9.6\times10^{-4}$ |
| lid-driven cavity, centreline $\max\lvert\Delta u\rvert$ | $8.82\times10^{-4}$ | $1.43\times10^{-6}$ |
| backward-facing step (RANS), $\Delta p$ | 0.678 % | 0.075 % |
| 3-D interrupted-fin unit cell, $\Delta p$ | 1.5 % | 0.11 % |

The difference is largest where $V/a_P$ jumps hardest between neighbours — at a
singular corner, and in a RANS field where $\nu_{\mathrm{eff}}$ spans orders of
magnitude. **The default remains $\gamma = 1$**, which is the consistent choice
for a solver that has no relaxation factor; $\gamma = \alpha_U$ is a verification
setting.

---

## 3. Collocated grids and Rhie-Chow interpolation

### 3.1 The checkerboard problem

All fields are stored at cell centres. If the face velocity needed by the
continuity equation is obtained by linear interpolation,
$\mathbf{u}_f = w_f\mathbf{u}_P + (1-w_f)\mathbf{u}_N$, and the pressure gradient
in the momentum equation by Green-Gauss with the same interpolation, then on a
uniform one-dimensional mesh of spacing $h$ the composed operator
$\mathrm{D}\,\mathrm{F}^{-1}\mathrm{G}$ acting on pressure is

$$
\left(\mathrm{D}\mathrm{F}^{-1}\mathrm{G}\,\mathbf{p}\right)_P
\;\propto\; \frac{p_{EE} - 2p_P + p_{WW}}{(2h)^2},
$$

a Laplacian on a stencil of width $2h$. It never sees $p_E$ or $p_W$. Its null
space therefore contains the mode $p_i = (-1)^i$ — a pressure field oscillating
cell to cell that produces no velocity at all. The discrete system fails the
inf-sup condition: pressure is not determined, and in practice the solution
develops a checkerboard.

### 3.2 The pseudo-momentum face-velocity reconstruction

Rhie & Chow (1983) removed the mode by reconstructing the face velocity from the
*momentum equation itself* rather than by interpolating the cell velocity. The
cell-centred momentum relation is

$$
\mathbf{u}_P \;=\; \frac{\mathbf{H}(\mathbf{u})_P}{a_P} \;-\; D_P\,(\nabla p)_P ,
\qquad D_P \;=\; \gamma\,\frac{V_P}{a_P}.
$$

Assume the same relation holds at the face, with $\mathbf{H}/a_P$ and $D$
interpolated but the pressure gradient evaluated **compactly**, i.e. from the two
cells straddling the face:

$$
\mathbf{u}_f \;=\; \left(\frac{\mathbf{H}(\mathbf{u})}{a_P}\right)_{\!f}
\;-\; D_f\,(\nabla p)^{\mathrm{compact}}_f .
$$

Substituting $\bigl(\mathbf{H}/a_P\bigr)_f = \bigl(\mathbf{u} + D\nabla
p\bigr)_f$ gives the working form

$$
\mathbf{u}_f \;=\; \left(\mathbf{u}\right)_f
\;+\; D_f\Bigl[(\nabla p)_f - (\nabla p)^{\mathrm{compact}}_f\Bigr]
$$

The correction is the difference between the wide-stencil (interpolated) pressure
gradient and the compact one. It vanishes to leading order for smooth pressure
fields and is maximal for the checkerboard mode, which is precisely the desired
behaviour: a fourth-difference dissipation acting on pressure.

### 3.3 The face flux this solver uses

Taking the dot product with $\mathbf{S}_f$ and writing the compact gradient as
$\Delta_f\,(p_N - p_P)$ along $\mathbf{d}_f$, the flux assembled and reconstructed
by this solver is

$$
\phi_f \;=\;
\underbrace{\bigl(w_f\mathbf{u}_P + (1-w_f)\mathbf{u}_N\bigr)\cdot\mathbf{S}_f}_{\text{(a) linear interpolation}}
\;-\;\underbrace{D_f\,\lvert\mathbf{S}_f\rvert\,\Delta_f\,(p_N - p_P)}_{\text{(b) compact normal gradient}}
\;+\;\underbrace{D_f\,(\nabla p)_f\cdot\mathbf{S}^{\mathrm{eff}}_f}_{\text{(c) wide-stencil gradient}}
$$

with

$$
D_f = \bigl(\text{interpolate}\bigr)\!\left[\gamma\,\frac{V_P}{a_P}\right],
\qquad
\mathbf{S}^{\mathrm{eff}}_f = \lvert\mathbf{S}_f\rvert\,\Delta_f\,\mathbf{d}_f ,
\qquad
(\nabla p)_f = \bigl(\text{interpolate}\bigr)\bigl[\nabla p\bigr] .
$$

$a_P$ is the coupled momentum diagonal: `UEqn.diag()` plus the patch
`internalCoeffs` accumulated onto their face cells and component-averaged — the
same quantity `fvMatrix::A()` returns — plus the pseudo-transient contribution
when it is active (§6.3).

**Why $\mathbf{S}^{\mathrm{eff}}$ and not $\mathbf{S}_f$.** OpenFOAM's corrected
Laplacian splits the face area vector as
$\mathbf{S}_f = \mathbf{S}^{\mathrm{eff}}_f + \bigl(\mathbf{S}_f -
\mathbf{S}^{\mathrm{eff}}_f\bigr)$, treating the first part implicitly through
$\lvert\mathbf{S}_f\rvert\Delta_f$ and the second explicitly through the
non-orthogonal correction vector $\hat{\mathbf{n}}_f - \Delta_f\mathbf{d}_f$.
Choosing $\mathbf{S}^{\mathrm{eff}}_f$ in term (c) makes the split *exact*:
expanding the standard segregated flux
$\phi = \bigl(\mathbf{u} + D\nabla p\bigr)_f\!\cdot\!\mathbf{S}_f - \bigl[\,
D\lvert\mathbf{S}_f\rvert\Delta_f(p_N-p_P) + D(\nabla p)_f\!\cdot\!(\mathbf{S}_f
- \mathbf{S}^{\mathrm{eff}}_f)\bigr]$ reproduces the expression above term for
term. On an orthogonal mesh $\mathbf{S}^{\mathrm{eff}}_f = \mathbf{S}_f$ and
terms (b) and (c) reduce to the classic compact-minus-wide gradient difference.

**Order of magnitude.** On a uniform one-dimensional mesh, a Taylor expansion of
(b) $+$ (c) about the face gives

$$
\phi_f^{(\mathrm{b})+(\mathrm{c})}
= -\,D_f\,h\left[\frac{h^2}{24} - \frac{7h^2}{24}\right]\frac{\partial^3 p}{\partial x^3}
+ O(h^5)
= \frac{D_f\,h^3}{4}\,\frac{\partial^3 p}{\partial x^3} + O(h^5),
$$

so the Rhie-Chow term contributes a third-derivative pressure flux. Relative to
the leading convective flux this is $O(h^2)$, which is the formal statement
behind §2.3: rescaling $D_f$ by $\gamma$ (or by $\alpha_U$) changes the converged
solution by an $O(h^2)$ term and by nothing else.

**The single documented formulation difference from OpenFOAM's segregated path**
is that `phiHbyA` uses `fvc::flux(rAU*grad(p))`, i.e. $r_{A_U}$ *inside* the
interpolation, whereas this solver uses $D_f\,(\nabla p)_f\cdot
\mathbf{S}^{\mathrm{eff}}_f$ with $D_f$ outside. The two agree to $O(h^2)$ and
differ most where $V/a_P$ jumps between neighbours; the residual
$9.6\times10^{-4}$ corner difference in the table of §2.3 is this term.

---

## 4. The coupled system

### 4.1 Unknown ordering and global indexing

Unknowns are interleaved per cell. The global row (and column) index of component
$i$ of cell $c$ is

$$
\mathrm{row}(c, i) \;=\; 4\,g(c) + i,
\qquad
i = 0,1,2 \rightarrow u,v,w; \quad i = 3 \rightarrow p,
$$

where $g(c)$ is the global cell index from `Foam::globalIndex` over
`mesh.nCells()`. The system has $4N_{\mathrm{global}}$ rows.

Two consequences follow from the interleaving, and both are load-bearing:

* The assembly emits backend-agnostic COO triplets
  $(\mathrm{row}, \mathrm{col}, \mathrm{value})$ carrying **global** indices.
  Each rank contributes only triplets whose row it owns; columns may point at
  cells owned by other ranks, reached through coupled patches. The identical
  assembly therefore feeds a serial direct backend and a distributed Krylov
  backend, and in serial the global indices equal the local ones, so the serial
  path pays nothing.
* The matrix has a genuine block size of 4. Declaring it
  (`MatSetBlockSize(A, 4)`) is what lets a field-split preconditioner derive the
  $[u,v,w]$ / $[p]$ index sets from options alone, with nothing coded (§8.3).

### 4.2 Block structure

Let $C_f = D_f\lvert\mathbf{S}_f\rvert\Delta_f$ and let
$\lambda_{f,P} = w_f$ when $P$ owns $f$ and $1-w_f$ otherwise. For an internal
face $f$ joining $P$ and $N$, the four $4\times4$ blocks written by the assembly
are

$$
\mathrm{K}_{PN}^{(f)} =
\begin{pmatrix}
a^{\mathrm{u}}_f & 0 & 0 & (1-w_f)S_{f,x}\\
0 & a^{\mathrm{u}}_f & 0 & (1-w_f)S_{f,y}\\
0 & 0 & a^{\mathrm{u}}_f & (1-w_f)S_{f,z}\\
(1-w_f)S_{f,x} & (1-w_f)S_{f,y} & (1-w_f)S_{f,z} & -C_f
\end{pmatrix},
$$

$$
\mathrm{K}_{NP}^{(f)} =
\begin{pmatrix}
a^{\mathrm{l}}_f & 0 & 0 & -w_f S_{f,x}\\
0 & a^{\mathrm{l}}_f & 0 & -w_f S_{f,y}\\
0 & 0 & a^{\mathrm{l}}_f & -w_f S_{f,z}\\
-w_f S_{f,x} & -w_f S_{f,y} & -w_f S_{f,z} & -C_f
\end{pmatrix},
$$

with $a^{\mathrm{u}}_f$, $a^{\mathrm{l}}_f$ the `upper()` and `lower()`
coefficients of the momentum matrix, and diagonal contributions
$+w_fS_{f,c}$ / $+C_f$ into $\mathrm{K}_{PP}$ and
$-(1-w_f)S_{f,c}$ / $+C_f$ into $\mathrm{K}_{NN}$ in the same positions. The
assembled diagonal block of cell $P$ is

$$
\mathrm{K}_{PP} =
\begin{pmatrix}
a_P^{x} & 0 & 0 & g^x_P\\
0 & a_P^{y} & 0 & g^y_P\\
0 & 0 & a_P^{z} & g^z_P\\
g^x_P & g^y_P & g^z_P & c_P
\end{pmatrix},
\qquad
\begin{aligned}
a_P^{c} &= a_P^{\mathrm{ldu}} + \textstyle\sum_{b} \mathrm{IC}^c_b + V_P/\Delta\tau_P,\\
g^c_P &= \textstyle\sum_{f\in\partial P} s_{f,P}\,\lambda_{f,P}\,S_{f,c},\\
c_P &= \textstyle\sum_{f\in\partial P} C_f .
\end{aligned}
$$

Three structural facts are worth reading off this:

1. **The velocity components are not coupled to each other.** `UEqn.diag()`,
   `upper()` and `lower()` are scalars applied identically to $u$, $v$ and $w$;
   only the per-component `internalCoeffs` differ. The $3\times3$ velocity
   sub-block is therefore diagonal. What this solver couples implicitly is
   $\mathbf{u}\leftrightarrow p$, not $u\leftrightarrow v$. Cross-component
   coupling exists only through the explicit $\mathrm{dev}_2$ stress and through
   boundary conditions.
2. **$\mathrm{G} = \mathrm{D}^{\mathsf{T}}$ on internal and coupled faces.** The
   coefficient linking the momentum row of $P$ to the pressure column of $N$ and
   the coefficient linking the continuity row of $P$ to the velocity column of
   $N$ are the same number. The discrete gradient is the exact transpose of the
   discrete divergence — a useful check when reading the assembly, and the reason
   the block system has the canonical saddle-point form of §1.3. Boundary patch
   terms break the symmetry, since the two use different coefficient sets (§5).
3. **$\mathrm{C}$ has positive diagonal and negative off-diagonals.** It is the
   negative Laplacian $-\nabla\cdot(D\nabla\,\cdot)$ in compact form: an
   M-matrix, symmetric positive definite up to the pressure level. This block —
   present only because of Rhie-Chow — is what converts a pure saddle point into
   a *stabilised* one.

### 4.3 Term-by-term derivation

#### (a) Momentum rows

The momentum operator is built from an **unrelaxed** matrix:

```cpp
fvVectorMatrix UEqn
(
    fvm::div(phi, U)
  + momentumTransport->divDevSigma(U)
);
```

This is the stock `incompressibleFluid` momentum predictor without `ddt`
(steady), without MRF and without `fvModels`, and without equation relaxation —
a coupled solve has no relaxation factor. Its coefficients are transferred as
follows.

| piece | source | destination |
|---|---|---|
| diagonal | `UEqn.diag()[c]` | $(4g(c)+i,\;4g(c)+i)$, $i=0,1,2$ |
| off-diagonals | `UEqn.upper()[f]`, `UEqn.lower()[f]` with `lduAddr().lowerAddr()`/`upperAddr()` | $(P,N)$ and $(N,P)$ |
| explicit source | `UEqn.source()[c][i]` | right-hand side |
| patch diagonal | `UEqn.internalCoeffs()[patch][i]` | diagonal, on **all** patches |
| patch source, uncoupled patch | `UEqn.boundaryCoeffs()[patch][i]` | right-hand side |
| patch link, coupled patch | $-$`UEqn.boundaryCoeffs()[patch][i]` | column of the **remote** cell |

The pressure gradient is added implicitly, discretised by Green-Gauss with the
mesh interpolation weights (not $\tfrac12$):

$$
V_P\,(\nabla p)_P \;=\; \sum_{f\in\partial P} s_{f,P}\,p_f\,\mathbf{S}_f ,
\qquad p_f = w_f p_P + (1-w_f)p_N ,
$$

so component $c$ of the momentum row of $P$ receives $+w_f S_{f,c}$ in the
pressure column of $P$ and $+(1-w_f)S_{f,c}$ in the pressure column of $N$, with
the signs reversed in the row of $N$. On boundary patches the face value comes
from the pressure field's own coefficients (§5.2), which makes the block exact
for every boundary-condition type that `fvc::grad` handles and makes
cyclic/processor support fall out for free.

#### (b) Continuity rows

The continuity row of cell $P$ is the signed sum of the Rhie-Chow fluxes of §3.3
over its faces:

$$
\sum_{f\in\partial P} s_{f,P}\,\phi_f \;=\; 0 .
$$

Term (a) of $\phi_f$ is linear in the unknown velocities and goes into the matrix
($\mathrm{D}$ block). Term (b) is linear in the unknown pressures and goes into
the matrix ($\mathrm{C}$ block). Term (c) contains $(\nabla p)_f$, evaluated from
the pressure field of the **previous** outer iteration, and goes to the
right-hand side:

$$
b_P^{\mathrm{cont}} \;=\; -\sum_{f\in\partial P} s_{f,P}\,D_f\,(\nabla p)_f\cdot\mathbf{S}^{\mathrm{eff}}_f .
$$

### 4.4 Implicit and lagged terms

| term | treatment | why |
|---|---|---|
| momentum convection $\nabla\cdot(\phi\mathbf{u})$ | implicit in $\mathbf{u}$, **lagged in $\phi$** | Picard linearisation (§6.1) |
| momentum diffusion, orthogonal part | implicit | `fvm::laplacian` |
| momentum diffusion, non-orthogonal correction | lagged (inside `UEqn.source()`) | standard OpenFOAM treatment |
| deviatoric stress $\nabla\cdot(\nu_{\mathrm{eff}}\,\mathrm{dev}_2[(\nabla\mathbf{u})^{\mathsf{T}}])$ | lagged (explicit source) | supplied by `divDevSigma`; would fill the velocity block |
| pressure gradient in momentum | **implicit**, full Green-Gauss | this is the coupling |
| face-velocity flux in continuity | **implicit** | this is the coupling |
| pressure Laplacian in continuity, orthogonal part | **implicit** | the stabilising $\mathrm{C}$ block |
| Rhie-Chow wide-stencil gradient, term (c) | **lagged** | see below |
| $\nu_t$ (and $k$, $\varepsilon$, $\omega$) | lagged one outer iteration | §7 |

**Why term (c) is lagged and not made implicit.** Term (c) is
$D_f(\nabla p)_f\cdot\mathbf{S}^{\mathrm{eff}}_f$ with $(\nabla p)_f$ the
interpolated *cell* gradient. Making it implicit would add a wide-stencil
Laplacian to the pressure block — the same $2h$ operator whose null space is the
checkerboard mode (§3.1). The implicit block would then be the near-cancellation
$\mathrm{C} - \mathrm{L}^{\mathrm{wide}}$, an $O(h^2)$ operator with a
checkerboard null space, and the very stabilisation Rhie-Chow was introduced to
provide would be destroyed inside the matrix. Lagging term (c) leaves the
implicit pressure block as the compact M-matrix $\mathrm{C}$ and pushes the
wide-stencil part into the outer iteration, where it converges rapidly (§6.5).

**Why the lagged split preserves discrete mass conservation exactly.** After each
linear solve the flux field is rebuilt from *the same expression*, with the same
$D_f$, the same $\Delta_f$, the same weights and the same lagged
$(\nabla p)_f$ that went into the matrix. Because $\phi_f$ is affine in
$\mathbf{x}$ and the assembly split it into an implicit part (the matrix row) and
a lagged part (the right-hand side) with no other change,

$$
\sum_{f\in\partial P} s_{f,P}\,\phi_f(\mathbf{x})
\;=\; \bigl(\mathrm{K}\mathbf{x}\bigr)_P^{\mathrm{cont}} - b_P^{\mathrm{cont}},
$$

i.e. **the discrete divergence of the reconstructed flux in each cell is exactly
the linear-solver residual of that cell's continuity row**. It is zero to the
tolerance of the inner solve — machine precision for a direct factorisation —
and this holds at *every* outer iteration, converged or not. The gradient is
refreshed from the new pressure only afterwards, for the next iteration.

The measured value on a lid-driven cavity at the first outer iteration is
$\sum_P\lvert\nabla\!\cdot\!\phi\rvert_P = 1.28\times10^{-16}$ with
$\max_P\lvert\nabla\!\cdot\!\phi\rvert_P = 1.64\times10^{-18}$, and it stays at
that level for every subsequent iteration and for turbulent cases with a strongly
varying $\nu_{\mathrm{eff}}$. Each iteration prints

```
coupled assembly check: sum|div(phi)| = ..., max|div(phi)| = ..., normalised = ...
```

so the property can be checked on any case.

---

## 5. Boundary conditions

Boundary conditions are never special-cased by type. Every patch contribution is
obtained from the coefficient interface that the boundary condition itself
implements, which is what makes the assembly exact for arbitrary boundary
condition types and makes processor and cyclic patches work without extra code.

### 5.1 Momentum rows

For a patch face with face cell $P$, OpenFOAM's `fvMatrix` convention is that the
patch contributes `internalCoeffs` to the diagonal and `boundaryCoeffs` to the
source. The assembly follows it literally: `internalCoeffs` is added to the
momentum diagonal on **all** patches, and `boundaryCoeffs` goes to the right-hand
side on uncoupled patches.

### 5.2 The pressure-gradient block

The face pressure needed by the Green-Gauss gradient is written through the
pressure field's own value coefficients,

$$
p_f \;=\; \mathrm{VIC}_b\,p_P \;+\; \mathrm{VBC}_b ,
$$

`VIC` = `valueInternalCoeffs(w)`, `VBC` = `valueBoundaryCoeffs(w)`. The momentum
row receives $+S_{f,c}\,\mathrm{VIC}_b$ in its own pressure column and
$-S_{f,c}\,\mathrm{VBC}_b$ on the right-hand side. For `fixedValue` pressure
$\mathrm{VIC}=0$, $\mathrm{VBC}=p_b$, so the whole term is known and moves to the
right-hand side; for `zeroGradient` $\mathrm{VIC}=1$, $\mathrm{VBC}=0$, so it
lands on the owner's pressure diagonal. Every intermediate case (`mixed`,
`inletOutlet`, ...) is handled by the same two lines.

### 5.3 The continuity row on a boundary

The velocity part uses the velocity field's value coefficients component-wise,
$u_{f,c} = \mathrm{uVIC}_c\,u_{P,c} + \mathrm{uVBC}_c$; the implicit part enters
the matrix and the known part is moved to the right-hand side with a minus sign.

The pressure part uses the pressure field's **gradient** coefficients, which
express the face-normal derivative as
$\;(\nabla p\cdot\hat{\mathbf{n}})_b = \mathrm{GIC}_b\,p_P + \mathrm{GBC}_b$.
The boundary flux term $-D_f\lvert\mathbf{S}_f\rvert(\mathrm{GIC}_b p_P +
\mathrm{GBC}_b)$ therefore contributes $-D_f\lvert\mathbf{S}_f\rvert\mathrm{GIC}_b$
to the owner's pressure diagonal and $+D_f\lvert\mathbf{S}_f\rvert\mathrm{GBC}_b$
to the right-hand side. For a `fixedValue` pressure patch,
$\mathrm{GIC}_b = -\Delta_b$ and $\mathrm{GBC}_b = \Delta_b p_b$, giving
$+D_f\lvert\mathbf{S}_f\rvert\Delta_b$ on the diagonal and
$+D_f\lvert\mathbf{S}_f\rvert\Delta_b p_b$ on the right-hand side — the standard
outlet term, and the reason a domain with a Dirichlet pressure outlet needs no
pinning. A `zeroGradient` pressure patch contributes nothing.

### 5.4 Fixed-flux patches

The Rhie-Chow pressure terms are switched off on any patch where

```cpp
U.boundaryField()[patchi].fixesValue()
```

is true. The physical argument is that Rhie-Chow exists to *reconstruct* an
unknown face velocity; where the velocity is prescribed the flux is exactly
known, and adding a pressure-difference correction to it would both falsify the
flux and couple the pressure level to a boundary that has no business setting it.
Consequently:

* **no-slip walls** contribute nothing at all ($\mathrm{uVIC}=\mathrm{uVBC}=0$
  and no pressure term);
* **velocity inlets** contribute only their known flux $\mathbf{u}_b\cdot
  \mathbf{S}_f$, moved to the right-hand side.

### 5.5 Impermeable constraint patches: the exact zero-flux rule

Symmetry planes, wedges and slip walls carry **exactly** zero mass flux. They do
not report `fixesValue()`, so without special handling they would be assembled
through the generic value-coefficient path of §5.3 — and that path cannot
represent zero flux on a face that is not axis-aligned.

The reason is worth spelling out, because the failure is silent and produces a
plausible-looking wrong answer. For a symmetry patch the exact boundary velocity
is the internal value with its normal component removed,

$$
\mathbf{u}_b = \left(\mathrm{I} - \hat{\mathbf{n}}\hat{\mathbf{n}}\right)\cdot\mathbf{u}_P ,
\qquad\text{so}\qquad
\phi_b = \mathbf{S}_f\cdot\mathbf{u}_b
= \lvert\mathbf{S}_f\rvert\left[\hat{\mathbf{n}}\cdot\mathbf{u}_P - (\hat{\mathbf{n}}\cdot\mathbf{u}_P)\right] = 0
$$

identically. But `valueInternalCoeffs` on a transform patch returns only the
**diagonal** of that transform tensor, i.e. the per-component coefficients
$1 - n_c^2$, discarding the off-diagonal $-n_c n_d$ terms. The flux the generic
path would build is

$$
\tilde\phi_b = \sum_c S_{f,c}\,(1-n_c^2)\,u_{P,c}
= \lvert\mathbf{S}_f\rvert\sum_c n_c\left(1-n_c^2\right)u_{P,c},
$$

which vanishes for every velocity field **if and only if** every $n_c \in
\{0,\pm 1\}$ — that is, only on an axis-aligned face. On a tilted face (any
`snappyHexMesh` symmetry plane) the leak is $O(1)$: the coefficient
$n_c(1-n_c^2)$ peaks at $0.385$ for $\hat{\mathbf{n}} = (1,1,1)/\sqrt3$. The
segregated algorithm never meets this problem because it evaluates the patch
value with the full transform tensor rather than with per-component coefficients.

The solver therefore classifies such patches explicitly,

```cpp
zeroFluxPatch(patchi) =  isA<symmetryFvPatch>(patch)
                      || isA<symmetryPlaneFvPatch>(patch)
                      || isA<wedgeFvPatch>(patch)
                      || isA<slipFvPatchVectorField>(U.boundaryField()[patchi]);
```

and forces both a zero continuity contribution and a zero reconstructed flux on
them, treating them like walls. On the 3-D interrupted-fin unit cell this moved
the pressure drop from 9.6 % low to 1.5 %, with the remainder attributable to the
Rhie-Chow damping definition of §2.3 (0.11 % once matched).

Note that the *momentum* rows on these patches are untouched: they still receive
the patch `internalCoeffs`/`boundaryCoeffs` and the implicit pressure-gradient
term. Only the continuity row and the flux reconstruction are overridden. The
transform coefficients of a **wedge** patch are likewise not accounted for in the
pressure-gradient block; wedge (axisymmetric) cases are consequently listed as
untested in the limitations.

### 5.6 Coupled patches and the sign convention

Processor and cyclic patches are assembled exactly like internal faces whose
neighbour happens to live elsewhere. The global index of the far-side cell comes
from `syncTools::swapBoundaryCellList(mesh, gCell, nbrGCell)`, which returns the
remote cell value for every boundary face and handles processor *and* cyclic
patches uniformly. The same call supplies remote $\mathbf{U}$ and $p$ for the
flux reconstruction and for the residual matrix-vector product.

The momentum contribution is

$$
\bigl(\mathrm{K}\mathbf{x}\bigr)_P \ \ni\ \mathrm{IC}\,u_{P} \;-\; \mathrm{BC}\,u_{R},
$$

i.e. `internalCoeffs` on the diagonal and $-$`boundaryCoeffs` in the remote
column. The minus sign is the convention of
`processorFvPatchField::updateInterfaceMatrix`, which computes
`result[faceCells] -= coeffs*pnf`. Getting it wrong does not produce a small
error — it produces a qualitatively different solution. The empirical proof that
the signs and the remote index exchange are correct is that parallel runs
reproduce the serial direct-solve results to $10^{-10}$ or better across
decompositions and rank counts, with **identical** outer iteration counts.

### 5.7 Pressure-level pinning

If any pressure patch reports `fixesValue()`, the pressure level is set by that
boundary and nothing is pinned. Otherwise the **continuity** row of global cell
`pRefCell` is replaced by the identity row $p_{\mathrm{ref}} = $ `pRefValue`. The
momentum rows of that cell are left untouched.

The pinned cell's continuity equation is thereby *not* enforced, which raises the
question of whether mass is conserved there. It is, for a consistent problem: the
remaining $N-1$ continuity rows are satisfied exactly (§4.4), the internal faces
cancel pairwise in the sum of all cell divergences, and the net boundary flux of a
closed or flux-balanced domain is zero, so the pinned cell's divergence is zero as
well. This is why the printed diagnostic reports the same value for the "full" and
"excluding pRef cell" sums to roundoff.

The corollary is that a domain whose prescribed inflow and outflow do **not**
balance produces a singular or inconsistent continuity block. No `adjustPhi`-style
rescaling is applied.

### 5.8 Empty patches and empty directions

Empty patches have `fvPatch::size() == 0`, so every patch loop skips them without
special-casing. Empty *directions* — components $c$ with `mesh.solutionD()[c] ==
-1` — receive an identity momentum row $u_c = 0$ and are excluded from the
gradient and continuity blocks, matching `fvMatrix::solveSegregated`, which skips
invalid components.

---

## 6. Solution algorithm

### 6.1 The outer Picard loop

One outer iteration performs, in order:

1. assemble `UEqn` from the current mass flux $\phi$ and the current
   $\nu_{\mathrm{eff}}$;
2. form $a_P$ (diagonal $+$ patch `internalCoeffs` $+$ pseudo-transient term),
   then $D_f$ and the lagged $(\nabla p)_f$;
3. assemble the coupled matrix and right-hand side, and pin the pressure if
   required;
4. compute the block residuals of the newly assembled system against the current
   solution, and test convergence;
5. solve the linear system with the current solution as the initial guess;
6. write $\mathbf{U}$ and $p$ back and correct their boundary conditions;
7. reconstruct $\phi$ from the same expression as the assembly;
8. call `viscosity->correct()` and `momentumTransport->correct()`, which run the
   segregated turbulence update on the just-reconstructed conservative flux;
9. read back the segregated fields' initial residuals for the next iteration's
   convergence test.

The only nonlinearity retained implicitly is none: the convective term is
linearised as $\nabla\cdot(\phi^{n}\mathbf{u}^{n+1})$ with the mass flux frozen
at the previous iteration. This is a **Picard** (fixed-point) linearisation, not
a Newton one — there is no $\partial(\mathbf{u}\otimes\mathbf{u})/\partial
\mathbf{u}$ Jacobian contribution. The outer loop therefore converges linearly,
not quadratically, and the count is governed by the contraction factor analysed
in §6.5.

### 6.2 Residual definition and normalisation

Residuals are OpenFOAM-style **initial** residuals of the assembled coupled
system, evaluated *before* the solve, so they measure both the linear and the
nonlinear (outer) error:

$$
\mathbf{r} = \mathbf{b}^{n+1} - \mathrm{K}^{n+1}\mathbf{x}^{n}.
$$

They are normalised per block component with the block generalisation of
`lduMatrix::solver::normFactor`. With $i = 4c + \mathrm{cmpt}$ running over the
rows of component $\mathrm{cmpt}$,

$$
\mathrm{res}_{\mathrm{cmpt}} \;=\;
\frac{\displaystyle\sum_{i}\bigl\lvert b_i - (\mathrm{K}\mathbf{x})_i\bigr\rvert}
{\max\left(\displaystyle\sum_{i}\Bigl[\bigl\lvert (\mathrm{K}\mathbf{x})_i - (\mathrm{K}\mathbf{x}_{\mathrm{ref}})_i\bigr\rvert
+ \bigl\lvert b_i - (\mathrm{K}\mathbf{x}_{\mathrm{ref}})_i\bigr\rvert\Bigr],\ \mathrm{vSmall}\right)}
$$

where $\mathbf{x}_{\mathrm{ref}}$ is the component-wise global average of
$\mathbf{x}$, held uniform across the mesh. All sums are reduced across ranks.
The log keeps the standard OpenFOAM format so that `foamLog` and PyFoam parse it
unchanged:

```
<solver>:  Solving for <field>, Initial residual = ..., Final residual = ..., No Iterations ...
```

**A known artefact of this normalisation**, inherited from stock OpenFOAM: when a
component's reference response $\mathrm{K}\mathbf{x}_{\mathrm{ref}}$ is
numerically zero and the right-hand side is too, the denominator collapses and the
reported *final* residual is meaningless. This is visible as
`Uy Final residual = 0.747` at the first iteration of a lid-driven cavity, where
the absolute linear residual is $\sim10^{-15}$. Convergence control uses only the
initial residuals, which are clean.

### 6.3 Pseudo-transient continuation

Optional (`pseudoTransient`, off by default). Per cell,

$$
h = V_P^{1/3}, \qquad
\Delta\tau_P = \mathrm{Co}_\tau\,\frac{h}{\max\bigl(\lvert\mathbf{u}_P\rvert,\;0.05\,U_{\mathrm{ref}}\bigr)},
\qquad
a_P \mathrel{+}= \frac{V_P}{\Delta\tau_P},
$$

with $V_P u^n_{P,c}/\Delta\tau_P$ added to the corresponding right-hand side.
$U_{\mathrm{ref}}$ is `pseudoUref` if positive, otherwise the global maximum of
$\lvert\mathbf{u}\rvert$ over internal cells and all boundary faces.

**It is exactly implicit under-relaxation with a locally computed factor.** Adding
$\beta_P = V_P/\Delta\tau_P$ to the diagonal and $\beta_P u^n$ to the source is
Patankar's implicit relaxation form with

$$
\alpha^{\mathrm{eq}}_P \;=\; \frac{a_P}{a_P + \beta_P}.
$$

Three consequences follow, and all three are observable:

* **It cancels at the fixed point.** When $\mathbf{u}^{n+1} = \mathbf{u}^{n}$ the
  added diagonal and the added source cancel identically, so the momentum balance
  of the converged solution is unaffected. It is a relaxation device, not a
  time-accurate scheme.
* **It does, however, change $D_f$.** Because $\beta_P$ is folded into $a_P$
  *before* $r_{A_U}$ is formed, $D_f = V_P/(a_P+\beta_P) =
  \alpha^{\mathrm{eq}}_P V_P/a_P$ — precisely the segregated relationship of
  §2.3. Enabling pseudo-transient continuation therefore shifts the converged
  answer by an $O(h^2)$ Rhie-Chow term of exactly the same family as
  `rhieChowFactor`. The measured spread across all converged configurations of the
  3-D interrupted-fin unit cell, with and without pseudo-transient continuation
  and across preconditioners and tolerances, is 0.04 %.
* **Large $\mathrm{Co}_\tau$ means weak damping.** With
  $a_P \sim \lvert\mathbf{u}\rvert h^2$ for a convection-dominated cell,
  $\beta_P/a_P \sim 1/\mathrm{Co}_\tau$, so the default
  $\mathrm{Co}_\tau = 200$ corresponds to $\alpha^{\mathrm{eq}}\approx0.995$.

**Why it makes inexact inner solves converge.** With a loose inner tolerance the
computed update is an approximation to the exact Picard step; the outer iteration
then amplifies whatever component of that error the fixed-point map amplifies.
The diagonal shift strengthens the velocity block (raising its diagonal dominance
and lowering $\lVert\mathrm{F}^{-1}\rVert$) and bounds the size of a single
update, so the composite map remains a contraction even when the linear solve
stops early. The control experiment on the 3-D unit cell is unambiguous: at
inner `relTol` $10^{-6}$ **with** damping the run converges in 21 outer
iterations, while **without** damping at $10^{-4}$ the residuals plateau at
$\sim6\times10^{-6}$ and never converge, despite each iteration being cheaper.

### 6.4 Convergence control

Convergence requires $\mathrm{iter} > 1$ and

$$
\mathrm{res}_{u},\ \mathrm{res}_{v},\ \mathrm{res}_{w} < \texttt{residualControl/U},
\qquad
\mathrm{res}_{p} < \texttt{residualControl/p},
$$

together with every additional `residualControl` entry, which is interpreted as a
**segregated** field whose initial residual is read back from OpenFOAM's own
per-time-step residual store through `convergenceControl::getInitialResiduals`.
Matching is on exact field names, not regular expressions, and fields that are
never solved are simply not checked. Those residuals are the ones measured at the
end of the *previous* outer iteration — lagged one iteration, like the turbulence
itself — so the reported converged iteration can be one later than a
same-iteration test would give.

### 6.5 Why the outer loop contracts at a mesh-independent rate

The following is a heuristic analysis, valid on a uniform orthogonal mesh with a
smooth pressure field and $\mathrm{F}\approx a_P\mathrm{I}$, but it predicts the
observed behaviour and explains what the outer loop is actually doing.

Write the pressure contribution to the continuity equation as
$\mathrm{L}^{\mathrm{c}} + \Lambda$, where $\mathrm{L}^{\mathrm{c}}$ is the
compact Laplacian of term (b) (implicit, in the matrix) and $\Lambda$ is the
wide-stencil operator of term (c) (lagged, on the right-hand side). By
construction the two nearly cancel in the smooth modes,
$\Lambda \approx -\mathrm{L}^{\mathrm{c}}$, which is the $O(h^2)$ statement of
§3.3. Eliminating velocity from the assembled system gives the Schur complement

$$
\mathrm{S}_{\mathrm{matrix}} = \mathrm{C} - \mathrm{D}\mathrm{F}^{-1}\mathrm{G}
\;\approx\; \mathrm{L}^{\mathrm{c}} + \mathrm{L}^{\mathrm{c}}
\;=\; 2\,\mathrm{L}^{\mathrm{c}},
$$

because $-\mathrm{D}\mathrm{F}^{-1}\mathrm{G}$ is itself a (wide-stencil)
discretisation of the same $-\nabla\cdot(D\nabla\,\cdot)$ operator. The
fully implicit Schur complement would instead be
$\mathrm{S}_{\mathrm{exact}} = (\mathrm{L}^{\mathrm{c}} + \Lambda) -
\mathrm{D}\mathrm{F}^{-1}\mathrm{G} \approx \mathrm{L}^{\mathrm{c}}$. The outer
iteration is therefore a stationary iteration whose amplification operator is

$$
-\,\mathrm{S}_{\mathrm{matrix}}^{-1}\Lambda \;\approx\; \tfrac12\,\mathrm{I}
$$

in the smooth pressure modes, and $\approx 0$ on the checkerboard mode, where
$\Lambda$ has no response. Three predictions follow:

* the contraction factor is $O(\tfrac12)$ and, crucially, **independent of $h$**,
  because both operators scale identically under refinement;
* the checkerboard mode is annihilated in a single outer iteration;
* the outer count does not depend on the number of ranks, since nothing in the
  argument refers to the decomposition.

All three are observed. Laminar cavities converge in **12** outer iterations at
$64^2$, $128^2$ and $256^2$ alike (an implied per-iteration factor of $\approx
0.39$ to reach $10^{-5}$), the two-dimensional interrupted-fin channel in 19-25,
and the 3-D interrupted-fin unit cell in 20-25. Iteration counts are *identical*,
not within $\pm1$, between serial and 4- and 6-rank runs of the same case. This is
the structural reason the coupled loop does not degrade with mesh refinement
whereas SIMPLE does: SIMPLE's amplification comes from dropping the off-diagonal
momentum coupling, whose relative weight grows with refinement, while the coupled
loop's comes from lagging a term that scales with the operator it approximates.

---

## 7. Turbulence

### 7.1 Why RANS is segregated in production coupled solvers

Turbulence is treated in the standard production way: **segregated and lagged one
outer iteration behind the coupled pressure-velocity solve**. This is what the
established coupled implementations do (Mangani, Buchmayr & Darwish 2014; Mangani, Darwish & Moukalled 2016;
ANSYS Fluent Theory Guide §23.4.3.2) and it is a deliberate choice rather than an
omission:

* **The stiff constraint is incompressibility, not turbulence.** Continuity is an
  instantaneous elliptic constraint with an infinite propagation speed in the
  discrete system; that is why segregating it costs so much. The turbulence field
  couples back to momentum only through the scalar $\nu_t$, and relaxes at a rate
  set by the physics of the transport equations, which no algebraic reordering
  can accelerate.
* **The extra equations would destroy the block structure the preconditioner
  relies on.** Extending the block to size 5-7 introduces equations that are
  strongly nonlinear in their own variables, require positivity clipping
  (`bounding k`), and carry wall functions that act on patch values rather than
  on matrix coefficients. The Schur-complement construction of §8 assumes a
  velocity block that behaves like a convection-diffusion operator; turbulence
  rows do not.
* **The measured payoff would be small.** The lag sets the asymptotic convergence
  rate for the *segregated* solver too (§7.4), so removing it from one solver and
  not the other would not change the comparison.

Pulling $k$ and $\varepsilon$ into the block system is documented as future work,
not as a scheduled change.

### 7.2 The momentum operator

```cpp
autoPtr<viscosityModel> viscosity(viscosityModel::New(mesh));
autoPtr<incompressible::momentumTransportModel> momentumTransport
(
    incompressible::momentumTransportModel::New(U, phi, viscosity)
);
momentumTransport->validate();
```

and

```cpp
fvVectorMatrix UEqn( fvm::div(phi, U) + momentumTransport->divDevSigma(U) );
```

`divDevSigma` is the method on the incompressible base class (an alias for
`divDevTau`) and is what the stock momentum predictor uses. Building the operator
from OpenFOAM's own framework rather than hand-coding a Laplacian means the entire
run-time selection table of incompressible laminar, RAS and LES models — and every
wall-function `fvPatchField` that comes with them — is available without further
code. In practice only `kEpsilon` and `kOmegaSST` have been exercised; the rest
are *available* rather than tested.

**For a laminar constant-viscosity model the operator is bit-for-bit the
hand-written one.** Unrolling the OpenFOAM 14 sources,

$$
\texttt{divDevSigma}(\mathbf{u}) =
\underbrace{\texttt{fvm::divc}\bigl(-\nu_{\mathrm{eff},f}\,(\mathbf{S}_f\cdot\mathrm{dev}_2[(\nabla\mathbf{u})^{\mathsf T}]),\,\mathbf{u}\bigr)}_{\text{explicit source only}}
\;-\; \texttt{fvm::laplacian}(\nu_{\mathrm{eff},f},\,\mathbf{u}),
$$

and three identities make it identical to the previous formulation: uniform-field
interpolation returns the field bitwise (OpenFOAM codes it as
$\lambda(v_P - v_N) + v_N$); `fvc::dotInterpolate(Sf, T)` equals
`Sf & fvc::interpolate(T)`; and `fvm::Su` writes only into `source()`, touching
neither `diag`, `upper`, `lower`, `internalCoeffs` nor `boundaryCoeffs`. The
regression is empirical as well as algebraic: two laminar cases on different
topologies produce logs that `diff` as empty against the pre-change binary over
12 and 25 outer iterations respectively, with a deterministic direct inner solve.

Note that `fvc::div` of a *surface* field is `surfaceIntegrate`, not a Gauss
scheme, so the explicit stress flux needs no `divSchemes` entry.

### 7.3 What is lagged, and where `correct()` sits

$k$, $\varepsilon$ / $\omega$, $\nu_t$ and hence $\nu_{\mathrm{eff}}$ are lagged
one outer iteration: the coupled system is assembled with the turbulence field
left from the previous iteration. `viscosity->correct()` and
`momentumTransport->correct()` are called **after** the flux reconstruction, so
the turbulence transport equations always run on the conservative flux the
coupled system just produced, never on a stale one, and $\nu_t$ is current for the
next assembly. `predict()` is deliberately not called — in OpenFOAM 14 the only
non-trivial overrides are in the generalised-Newtonian and laminar/Stokes chains,
all no-ops for eddy-viscosity RAS models. `validate()` *is* called after
construction; for `eddyViscosity` it is `correctNut()`, making $\nu_t$ consistent
with the initial $k$ and $\varepsilon$ before the first matrix is assembled.

Everything downstream of the operator is operator-agnostic and was unchanged when
turbulence support was added: coefficient extraction, $a_P$, $r_{A_U}$ and $D_f$,
the pseudo-transient diagonal, `rhieChowFactor`, the Rhie-Chow assembly, the flux
reconstruction, the zero-flux rule and the residual machinery.

### 7.4 The lag sets the convergence floor

This is the central practical consequence and it follows directly from the
residual definition of §6.2. The measured residual is
$\mathbf{r} = \mathbf{b}^{n+1} - \mathrm{K}^{n+1}\mathbf{x}^{n}$, and
$\mathrm{K}^{n+1}$ contains $\nu_{\mathrm{eff}} = \nu + \nu_t$ both in the
momentum coefficients and, through $a_P$, in $D_f$. **While $\nu_t$ changes, the
operator itself changes between iterations, and the residual cannot vanish
however exactly the pressure-velocity block is solved.**

RANS convergence consequently has two phases. Phase one is fast and is the
coupled solver doing its job: on a backward-facing step with `kEpsilon`, $u_x$
falls from 1 to $1.7\times10^{-3}$ in **61** outer iterations where SIMPLE needs
451. Phase two is a long plateau at $u_x \sim 5\times10^{-5}$, lasting hundreds of
iterations, which collapses by two orders only once $k$ and $\varepsilon$ reach
$\sim10^{-5}$.

The plateau is attributable to the lag and to nothing else. Restarting the plateaued
fields with `turbulence off` — freezing $\nu_t$, changing nothing else — makes the
coupled pressure-velocity block converge geometrically at $\approx 0.82$ per
iteration:

```
  it 401   Ux 3.95268e-05   p 0.000115441
  it 410   Ux 6.91999e-06   p 1.6755e-06
  it 420   Ux 1.04405e-06   p 2.64393e-07
  it 440   Ux 2.86898e-08   p 7.20118e-09
```

Forty iterations take the residuals from $4\times10^{-5}$ to $3\times10^{-8}$.
The same floor appears in the segregated solver, so it is a property of lagged
RANS rather than of this solver.

Against SIMPLE at matched residual levels the coupled solver retains a 4-7$\times$
advantage **per iteration** ($u_x < 1.7\times10^{-3}$: 61 versus 451;
$u_x < 3.2\times10^{-4}$: 181 versus 777), and 2.6$\times$ end to end (779 versus
1997). But each coupled iteration is far more expensive, so on small
two-dimensional RANS cases the segregated solver wins on wall clock. The coupled
solver's RANS niche is a deeper residual floor at equal compute — measured
4.7$\times$ deeper in $u_x$ on the 3-D interrupted-fin unit cell, where both
solvers lag-floor.

A second, milder consequence: because $\nu_{\mathrm{eff}}$ is heterogeneous in a
RANS field, both the velocity-block preconditioner and the Schur approximation
degrade, so the inner solve becomes markedly more expensive than in laminar cases
(§8.5).

---

## 8. Linear algebra

### 8.1 What the linear solver is handed

$$
\mathrm{K} = \begin{pmatrix} \mathrm{F} & \mathrm{G} \\ \mathrm{D} & \mathrm{C}\end{pmatrix},
\qquad \mathrm{G} = \mathrm{D}^{\mathsf{T}} \ \text{(interior)},
\qquad \mathrm{C} \succ 0 \ \text{compact, small}.
$$

$\mathrm{F}$ is a nonsymmetric convection-diffusion operator, block-diagonal in
the velocity components. $\mathrm{K}$ is nonsymmetric, indefinite, and — after the
sign normalisation of §1.3 — has a block-diagonal symmetric part with one positive
and one negative block. It is a *stabilised* saddle-point matrix in the sense of
Benzi, Golub & Liesen (2005), where the stabilisation $\mathrm{C}$ is small but
essential.

### 8.2 Why generic AMG fails on it

Classical and smoothed-aggregation algebraic multigrid rest on assumptions that
this matrix violates:

* **Point smoothers are not smoothers here.** Jacobi, Gauss-Seidel and point ILU
  reduce high-frequency error for definite, diagonally dominant operators. On an
  indefinite saddle-point matrix they do not reduce error in the
  near-divergence-free subspace at all, and the pressure rows have no diagonal at
  all before Rhie-Chow and only a small one afterwards.
* **The near-null space is not the constant vector.** AMG interpolation is built
  to represent the algebraically smooth error, which for a scalar elliptic
  operator is well represented by constants. For the coupled system it is a
  vector-valued, block-structured object, and the coarse operators must
  themselves satisfy an inf-sup condition to be solvable — which unmodified
  aggregation does not guarantee.

The measurements are correspondingly blunt. Smoothed-aggregation AMG applied to
the whole block system on a 483k-cell mesh was killed after 20 minutes without
completing one outer iteration. Scalar BoomerAMG on the whole system diverges at
the 5000-iteration inner cap. By contrast, hypre's **nodal** (systems) AMG, which
does respect the block structure, converges on the same problem to the
direct-solve reference at 2361 inner iterations — evidence that the obstacle is
the block/indefinite structure, not the size.

### 8.3 Field-split and Schur-complement preconditioning

The productive route is to precondition with the block factorisation

$$
\mathrm{K} =
\begin{pmatrix}\mathrm{I} & 0\\ \mathrm{D}\mathrm{F}^{-1} & \mathrm{I}\end{pmatrix}
\begin{pmatrix}\mathrm{F} & 0\\ 0 & \mathrm{S}\end{pmatrix}
\begin{pmatrix}\mathrm{I} & \mathrm{F}^{-1}\mathrm{G}\\ 0 & \mathrm{I}\end{pmatrix},
\qquad
\mathrm{S} = \mathrm{C} - \mathrm{D}\mathrm{F}^{-1}\mathrm{G},
$$

which is exact, and to replace $\mathrm{F}^{-1}$ and $\mathrm{S}^{-1}$ by
approximations (Elman, Silvester & Wathen 2014). This is `PCFIELDSPLIT` with
`-pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type full`. The $[u,v,w]$ /
$[p]$ index sets come free from the declared block size of 4 — nothing about the
split is coded, and everything is switchable from the `petscOptions` string.

The velocity block is handled by block-Jacobi with sub-domain ILU: adequate,
because $\mathrm{F}$ is a well-behaved convection-diffusion operator. Replacing it
with plain Jacobi nearly doubles the inner iteration count, and replacing its ILU
with a device-resident variant is worth $0.01$ s per outer iteration — the
velocity block is not where the time goes.

### 8.4 SIMPLE-type Schur approximations

$\mathrm{S}$ is dense and never formed. PETSc's `selfp` option approximates it as

$$
\widehat{\mathrm{S}} = \mathrm{C} - \mathrm{D}\,\mathrm{diag}(\mathrm{F})^{-1}\mathrm{G},
$$

which is **exactly the SIMPLE pressure-correction matrix**: the same operator
§2.1 obtains by dropping the neighbour terms of $\mathrm{H}$. The framing this
invites is accurate and useful — *a coupled solver preconditioned this way
contains SIMPLE as its preconditioner and wraps a Krylov method around it*, so
the approximation that forces SIMPLE to under-relax becomes an approximation that
merely costs the Krylov solver a few extra iterations.

Both $\mathrm{C}$ and $\mathrm{D}\,\mathrm{diag}(\mathrm{F})^{-1}\mathrm{G}$
discretise $-\nabla\cdot(D\nabla\,\cdot)$ — the first compactly, the second on a
wide stencil — and by the sign analysis of §6.5 they **add** rather than cancel in
$\widehat{\mathrm{S}}$. The Schur block handed to the inner solver is therefore a
scalar, elliptic, positive-definite, checkerboard-free Laplacian-like operator.
That is precisely the class of problem algebraic multigrid was built for, which is
why AMG is applied *to the Schur block* rather than to the whole system.

Two alternatives were measured and are worse here: the multiplicative field-split
variant reaches the same answer with no gain, and least-squares commutator (LSC)
preconditioning converges but at 211 s per outer iteration, roughly 40$\times$ the
target.

### 8.5 Block AMG in the literature, and why it is not used here

The alternative to splitting is a multigrid method built for the coupled system
directly: coarsening on the block matrix with block smoothers. The literature is
well developed — Uroic & Jasak (2018, 2021) on block-selective AMG for implicitly
coupled pressure-velocity systems; Darwish, Sraj & Moukalled (2009), Mangani,
Buchmayr & Darwish (2014) and Mangani, Darwish & Moukalled (2016) on coupled
finite-volume solvers with
purpose-built coupled AMG; aggregation-based block AMG as used in production
coupled solvers (arXiv:2403.07882); ICSFoam as an example of coupled AMG with
block-ILU smoothers; and Notay's pairwise/double-pairwise aggregation (Notay
2010) as an architecture with intrinsically cheap setup.

This solver does not implement one. It delegates to PETSc's generic components,
for which the honest reason is a measurement: **preconditioner reuse is
counterproductive here.** Freezing the preconditioner across outer iterations
costs $+53\%$ wall clock on the 483k-cell case (392.7 s against 257.3 s), and
freezing only the hierarchy structure costs $+66\%$. The Picard operator evolves
fast enough between iterations that a stale preconditioner costs more inner
iterations than its setup saves. Any block-AMG work would therefore have to make
*fresh setup intrinsically cheap* rather than amortise setup across iterations —
which is exactly the design point of pairwise aggregation, and exactly what the
smoothed-aggregation and classical hierarchies do not offer.

### 8.6 Preconditioner suitability is problem-dependent

This is the practically important conclusion, and it is not a hedge — the measured
behaviour is a clean mirror image:

| configuration | 2-D cavity-class | 2-D interrupted-fin strip channel | 3-D snappyHexMesh unit cell (1.93M unknowns) |
|---|---|---|---|
| ASM(overlap 2) / sub-ILU(2) | converges | **converges** | **caps out** — 5000 inner iterations, unusable |
| field-split Schur (`selfp`) + GAMG | converges, ~1.9$\times$ faster than ASM | **diverges**, serial and parallel | converges, 130-470 inner iterations |
| field-split Schur (`selfp`) + hypre BoomerAMG | converges | **converges** to the reference's every digit | **converges**, ~75 inner iterations |
| AMG on the whole block system | — | diverges (scalar) / converges slowly (nodal) | total failure |

Each of the first two is robust exactly where the other fails. Only field-split
Schur with hypre BoomerAMG on the Schur block passes every tested topology, which
is why it is the shipped default. Replacing GAMG with BoomerAMG on the Schur block
cut the inner iteration count from 188 to $\approx75$ and the cost from 19.95 to
7.72 s per outer iteration — a 2.7$\times$ end-to-end speed-up from a one-line
options change and no hardware.

The inner tolerance interacts with this choice and cannot be set once and
forgotten, because the outer loop is a Picard iteration started from a nonzero
guess: too loose an inner solve makes the linear update negligible and the outer
loop stalls rather than converging slowly.

| regime | recommended `relTol` | rationale |
|---|---|---|
| small laminar (2-D) | $10^{-12}$ | $10^{-3}$ and $10^{-6}$ both fail to converge |
| large laminar (3-D) with pseudo-transient | $10^{-6}$ | 1.9$\times$ faster than $10^{-12}$ at the same outer count; $10^{-4}$ costs more outer iterations than it saves |
| small RANS | $10^{-12}$ | at $10^{-6}$ discrete continuity degrades to $\sim10^{-9}$ and the outer loop does not finish |
| large 3-D RANS | $10^{-8}$ with `maxLinearIter 500` | at $10^{-12}$ the turbulent matrix needs ~2340 inner iterations per outer solve |

---

## 9. Relationship to other implementations

### 9.1 Segregated OpenFOAM

The segregated reference is `foamRun -solver incompressibleFluid` with SIMPLE
dictionaries; Foundation OpenFOAM 14 has no standalone `simpleFoam`. The two
solvers share the discretisation family exactly: the same convection and diffusion
schemes, the same non-orthogonal treatment, the same Green-Gauss gradient, the same
Rhie-Chow structure. They differ in three identified places:

1. **the path** — Picard on a coupled matrix versus SIMPLE with under-relaxation;
2. **$D_f$** — $V/a_P$ versus $\alpha_U V/a_P$, an $O(h^2)$ term reproducible with
   `rhieChowFactor` (§2.3);
3. **the flux gradient term** — $D_f(\nabla p)_f\cdot\mathbf{S}^{\mathrm{eff}}_f$
   versus `fvc::flux(rAU*grad(p))`, i.e. $r_{A_U}$ outside versus inside the
   interpolation, also $O(h^2)$ (§3.3).

Against a *properly converged* segregated reference the agreement is
$1.03\times10^{-5}$ on a cavity centreline; against a loosely converged one it is
$8.8\times10^{-4}$, because the reference's own residual level is the larger error.
Pressure-drop agreement across the laminar suite is 0.17-0.38 % at production
stopping criteria and 0.11 % on the 3-D unit cell once $D_f$ is matched.

The behavioural difference the coupled formulation buys is not primarily speed. It
is (i) no relaxation factors to tune and no divergence when they are set
aggressively, (ii) outer iteration counts that do not grow under mesh refinement,
and (iii) the ability to converge deeply on meshes where SIMPLE floors — on the
3-D interrupted-fin unit cell the segregated solver never reaches $10^{-7}$ in
8000 iterations, while the coupled solver reaches $8\times10^{-10}$ in 28.

### 9.2 foam-extend `pUCoupledFoam`

`pUCoupledFoam` is the freely available prior art for a block-coupled
pressure-velocity solver in the OpenFOAM family. It is built on foam-extend's
`BlockLduMatrix` infrastructure and its own block AMG, and is therefore confined
to the foam-extend fork; it cannot be used with Foundation OpenFOAM 14, whose
`fvMatrix` has no block-coupled equivalent. This solver takes the other route —
extract standard `fvMatrix` coefficients and assemble an external global matrix —
which keeps it inside mainline OpenFOAM at the cost of owning the linear algebra.
The published description of that solver is Jareteg, Vukcevic and Jasak (2014).

### 9.3 Commercial pressure-based coupled solvers

The pressure-based coupled scheme in ANSYS Fluent (Theory Guide §23.4.3.2) is the
same algebraic idea: momentum and continuity in one block system, pseudo-transient
continuation (User's Guide §37.3) in place of under-relaxation factors, and
turbulence segregated. The differences are in the linear algebra rather than the
formulation — a purpose-built coupled algebraic multigrid with block smoothers,
tuned over a long period against a broad case base, where this solver delegates to
generic PETSc components and compensates with Schur-complement splitting. That is
the honest explanation both for why the coupled formulation behaves as the
literature says it should here, and for why wall-clock parity with a mature
segregated solver is not yet reached (§8.6).

### 9.4 petsc4Foam and the PETSc external-solver module

petsc4Foam (Bnà et al., PRACE white paper WP294) plugs PETSc in as a drop-in
replacement for OpenFOAM's `lduMatrix` solvers, i.e. at the level of the
individual **segregated** equations. It does not change the algorithm, only who
solves each scalar system. This solver operates at a different level: it builds
its own global block matrix with global indices and hands PETSc a
$4N\times4N$ system that has no counterpart in the segregated data structures. The
two are complementary rather than alternatives, and this repository has no
petsc4Foam dependency.

### 9.5 Adaptive inner tolerances

The inner-tolerance regimes of §8.6 are a hand-tuned substitute for an adaptive
forcing-term strategy of the Eisenstat & Walker (1996) type, in which the inner
tolerance is chosen from the observed outer residual reduction. This is an
unexplored avenue for reducing inner-solve cost, and would directly address the
one control the user currently cannot set once and forget.

---

*Numerical values quoted in this document are measurements of this solver;
their full context, case definitions and stopping criteria are in
[`BENCHMARKS.md`](BENCHMARKS.md). All cited works are listed in
[`REFERENCES.md`](REFERENCES.md).*
