# References

Literature and software this solver is built on. The author-year keys used here
are the ones cited in `doc/THEORY.md`; each entry carries a one-line note on what
`coupledSimpleFoam` takes from it. DOIs are given where they exist, stable URLs
otherwise.

---

## 1. Finite volume method and pressure-velocity coupling

- **Moukalled, Mangani & Darwish (2016)**

  F. Moukalled, L. Mangani and M. Darwish. *The Finite Volume Method in
  Computational Fluid Dynamics: An Advanced Introduction with OpenFOAM and
  Matlab*. Fluid Mechanics and Its Applications. Springer, Cham, 2016.
  ISBN 978-3-319-16873-9.
  doi:[10.1007/978-3-319-16874-6](https://doi.org/10.1007/978-3-319-16874-6)

  The discretisation reference for the whole solver: collocated finite-volume
  treatment of the incompressible Navier-Stokes equations, Rhie-Chow momentum
  interpolation, and the presentation of the coupled momentum-continuity
  assembly whose block structure this implementation reproduces.

- **Patankar & Spalding (1972)**

  S. V. Patankar and D. B. Spalding. "A calculation procedure for heat, mass and
  momentum transfer in three-dimensional parabolic flows". *International Journal
  of Heat and Mass Transfer*, 15(10):1787-1806, 1972.
  doi:[10.1016/0017-9310(72)90054-3](https://doi.org/10.1016/0017-9310(72)90054-3)

  The original SIMPLE algorithm. It defines the segregated pressure-correction
  splitting that the coupled formulation removes, and the segregated solver used
  as the validation and performance baseline is a direct descendant of it.

- **Van Doormaal & Raithby (1984)**

  J. P. Van Doormaal and G. D. Raithby. "Enhancements of the SIMPLE method for
  predicting incompressible fluid flows". *Numerical Heat Transfer*,
  7(2):147-163, 1984.
  doi:[10.1080/01495728408961817](https://doi.org/10.1080/01495728408961817)

  SIMPLEC, and the analysis of why segregated schemes require under-relaxation
  factors at all. It is the reference point for the solver's central claim that
  no pressure-velocity under-relaxation is needed once the two equations are
  solved implicitly together.

- **Issa (1986)**

  R. I. Issa. "Solution of the implicitly discretised fluid flow equations by
  operator-splitting". *Journal of Computational Physics*, 62(1):40-65, 1986.
  doi:[10.1016/0021-9991(86)90099-9](https://doi.org/10.1016/0021-9991(86)90099-9)

  PISO, the other classical operator-splitting scheme. Cited as the second member
  of the splitting family whose per-iteration cheapness is paid for in outer
  iteration counts that grow with mesh refinement.

- **Rhie & Chow (1983)**

  C. M. Rhie and W. L. Chow. "Numerical study of the turbulent flow past an
  airfoil with trailing edge separation". *AIAA Journal*, 21(11):1525-1532, 1983.
  doi:[10.2514/3.8284](https://doi.org/10.2514/3.8284)

  Momentum interpolation for the face flux on a collocated grid. The continuity
  row of the block system is this flux made implicit in both the velocity and the
  pressure unknowns; the `rhieChowFactor` entry scales the damping term so that a
  run can be matched to a segregated reference that bakes its relaxation factor
  into the same coefficient.

---

## 2. Coupled pressure-based solvers

- **Darwish, Sraj & Moukalled (2009)**

  M. Darwish, I. Sraj and F. Moukalled. "A coupled finite volume solver for the
  solution of incompressible flows on unstructured grids". *Journal of
  Computational Physics*, 228(1):180-201, 2009.
  doi:[10.1016/j.jcp.2008.08.027](https://doi.org/10.1016/j.jcp.2008.08.027)

  The paper this implementation follows for the assembly: the interleaved
  `[u, v, w, p]` cell ordering, the implicit pressure-gradient block in the
  momentum rows, and the implicit velocity and pressure contributions to the
  continuity row.

- **Mangani, Buchmayr & Darwish (2014a)**

  L. Mangani, M. Buchmayr and M. Darwish. "Development of a novel fully coupled
  solver in OpenFOAM: steady-state incompressible turbulent flows". *Numerical
  Heat Transfer, Part B: Fundamentals*, 66(1):1-20, 2014.
  doi:[10.1080/10407790.2014.894448](https://doi.org/10.1080/10407790.2014.894448)

  Precedent for building a coupled pressure-velocity solver inside OpenFOAM with
  the turbulence equations left segregated and lagged one outer iteration behind
  the coupled solve, which is the treatment used here.

- **Mangani, Buchmayr & Darwish (2014b)**

  L. Mangani, M. Buchmayr and M. Darwish. "Development of a novel fully coupled
  solver in OpenFOAM: steady-state incompressible turbulent flows in rotational
  reference frames". *Numerical Heat Transfer, Part B: Fundamentals*,
  66(6):526-543, 2014.
  doi:[10.1080/10407790.2014.894372](https://doi.org/10.1080/10407790.2014.894372)

  The rotating-frame extension of the same coupled framework; the reference for
  the additional source and flux terms a future MRF or rotating-frame variant of
  this solver would need.

- **Mangani, Darwish & Moukalled (2016)**

  L. Mangani, M. Darwish and F. Moukalled. "An OpenFOAM pressure-based coupled
  CFD solver for turbulent and compressible flows in turbomachinery
  applications". *Numerical Heat Transfer, Part B: Fundamentals*, 69(5):413-431,
  2016.
  doi:[10.1080/10407790.2015.1125212](https://doi.org/10.1080/10407790.2015.1125212)

  Evidence that the same coupled assembly carries to industrial-scale
  applications, and the reference for how the pressure block changes once density
  is no longer constant - the natural next scope beyond this solver's
  incompressible limit.

- **Darwish & Moukalled (2014)**

  M. Darwish and F. Moukalled. "A fully coupled Navier-Stokes solver for fluid
  flow at all speeds". *Numerical Heat Transfer, Part B: Fundamentals*,
  65(5):410-444, 2014.
  doi:[10.1080/10407790.2013.869102](https://doi.org/10.1080/10407790.2013.869102)

  The all-speed generalisation of the coupled framework, including the
  pressure-density coupling terms; consulted for the boundary-condition
  linearisation of the coupled system.

- **Jareteg, Vukčević & Jasak (2014)**

  K. Jareteg, V. Vukčević and H. Jasak. "pUCoupledFoam - an open source coupled
  incompressible pressure-velocity solver based on foam-extend". 9th OpenFOAM
  Workshop, Zagreb, Croatia, 2014.
  [openfoam-extend.sourceforge.net/OpenFOAM_Workshops/OFW9_2014_Zagreb](https://openfoam-extend.sourceforge.net/OpenFOAM_Workshops/OFW9_2014_Zagreb/download.html)

  The reference open-source coupled pressure-velocity solver in the OpenFOAM
  family. It establishes the block-matrix route taken by foam-extend, against
  which this solver's alternative - emitting backend-agnostic COO triplets to an
  external linear-algebra library - is the deliberate design contrast.

- **Uroić & Jasak (2018)**

  T. Uroić and H. Jasak. "Block-selective algebraic multigrid for implicitly
  coupled pressure-velocity system". *Computers & Fluids*, 167:100-110, 2018.
  doi:[10.1016/j.compfluid.2018.02.034](https://doi.org/10.1016/j.compfluid.2018.02.034)

  Block-selective AMG for coupled pressure-velocity systems: coarsening driven
  by a selected equation set rather than by point-wise strength of connection.
  It defines what a purpose-built preconditioner for this matrix looks like, and
  the qualification carries weight, because the same architecture assembled from
  generic library components has been measured at 8.6 million unknowns and costs
  three to four times more per outer iteration than the Schur splitting it would
  replace (doc/BENCHMARKS.md section 9.3). The selective coarsening and block
  smoothers described here are therefore what a competitive implementation would
  have to be written out of, rather than an untried alternative to the shipped
  field-split configuration.

- **Uroić & Jasak (2021)**

  T. Uroić and H. Jasak. "Parallelisation of selective algebraic multigrid for
  block-pressure-velocity system in OpenFOAM". *Computer Physics
  Communications*, 258:107529, 2021.
  doi:[10.1016/j.cpc.2020.107529](https://doi.org/10.1016/j.cpc.2020.107529)

  The distributed-memory version of the same method, and the source of the
  expectation this solver checks in its parallel benchmarks: outer iteration
  counts that stay independent of the number of ranks.

- **Vendor documentation: pressure-based coupled algorithm and pseudo-transient
  continuation**

  ANSYS, Inc. *Ansys Fluent Theory Guide*, Release 2025 R2, section
  "Pressure-Based Coupled Algorithm"; and *Ansys Fluent User's Guide*, Release
  2025 R2, section "Pseudo Time Method". ANSYS, Inc., Canonsburg, PA, 2025.
  Distributed with the product and through the vendor's documentation portal.

  Commercial-solver documentation of the same algorithm class, cited for two
  things: the description of a pressure-based coupled scheme run without
  pressure-velocity under-relaxation, and pseudo-transient continuation - a local
  pseudo-time step derived from a target Courant number, added to the diagonal
  and vanishing at convergence. The latter is implemented here as the
  `pseudoTransient`, `pseudoCo` and `pseudoUref` controls.

---

## 3. Linear algebra and preconditioning

- **Benzi, Golub & Liesen (2005)**

  M. Benzi, G. H. Golub and J. Liesen. "Numerical solution of saddle point
  problems". *Acta Numerica*, 14:1-137, 2005.
  doi:[10.1017/S0962492904000212](https://doi.org/10.1017/S0962492904000212)

  The survey that classifies the assembled block system as a (stabilised) saddle
  point problem and explains why generic point-wise smoothers and plain
  aggregation multigrid fail on it - the behaviour observed when such
  preconditioners are applied to this matrix.

- **Elman, Silvester & Wathen (2014)**

  H. C. Elman, D. J. Silvester and A. J. Wathen. *Finite Elements and Fast
  Iterative Solvers: with Applications in Incompressible Fluid Dynamics*, 2nd
  edition. Oxford University Press, Oxford, 2014.
  doi:[10.1093/acprof:oso/9780199678792.001.0001](https://doi.org/10.1093/acprof:oso/9780199678792.001.0001)

  Schur-complement preconditioning theory for incompressible flow: block
  factorisations, the role of the pressure Schur complement, and the convergence
  arguments behind the field-split preconditioner used as the parallel default.

- **Klaij & Vuik (2013)**

  C. M. Klaij and C. Vuik. "SIMPLE-type preconditioners for cell-centered,
  colocated finite volume discretization of incompressible Reynolds-averaged
  Navier-Stokes equations". *International Journal for Numerical Methods in
  Fluids*, 71(7):830-849, 2013.
  doi:[10.1002/fld.3686](https://doi.org/10.1002/fld.3686)

  SIMPLE reinterpreted as a preconditioner rather than an outer iteration. This
  is the justification for the default Schur approximation, which builds the
  pressure operator from the velocity block's diagonal exactly as SIMPLE does.

- **Elman, Howle, Shadid, Shuttleworth & Tuminaro (2006)**

  H. Elman, V. E. Howle, J. Shadid, R. Shuttleworth and R. Tuminaro. "Block
  preconditioners based on approximate commutators". *SIAM Journal on Scientific
  Computing*, 27(5):1651-1668, 2006.
  doi:[10.1137/040608817](https://doi.org/10.1137/040608817)

  The least-squares commutator (LSC) Schur approximation, evaluated here as an
  alternative to the SIMPLE-type approximation: robust across the test
  topologies, but too expensive per outer iteration on three-dimensional meshes
  to be the default.

- **Elman, Howle, Shadid, Shuttleworth & Tuminaro (2008)**

  H. Elman, V. E. Howle, J. Shadid, R. Shuttleworth and R. Tuminaro. "A taxonomy
  and comparison of parallel block multi-level preconditioners for the
  incompressible Navier-Stokes equations". *Journal of Computational Physics*,
  227(3):1790-1808, 2008.
  doi:[10.1016/j.jcp.2007.09.026](https://doi.org/10.1016/j.jcp.2007.09.026)

  The comparison framework used when screening preconditioner configurations for
  this solver, and the reason the screening insists a configuration prove itself
  on more than one flow topology before becoming a default.

- **Dembo, Eisenstat & Steihaug (1982)**

  R. S. Dembo, S. C. Eisenstat and T. Steihaug. "Inexact Newton methods". *SIAM
  Journal on Numerical Analysis*, 19(2):400-408, 1982.
  doi:[10.1137/0719025](https://doi.org/10.1137/0719025)

  The origin of the inexact-Newton idea, and of the forcing term itself: the
  inner linear solve is asked only to reduce the residual of the current iterate
  by a factor, and the local convergence theory is what makes a sequence of
  deliberately loose inner solves legitimate rather than merely cheap. The outer
  loop here is Picard rather than Newton, so the theory does not carry over
  verbatim, but the quantity `adaptiveTolerance` sets is this forcing term and
  the convergence test it presumes is the one that paper defines.

- **Eisenstat & Walker (1996)**

  S. C. Eisenstat and H. F. Walker. "Choosing the forcing terms in an inexact
  Newton method". *SIAM Journal on Scientific Computing*, 17(1):16-32, 1996.
  doi:[10.1137/0917003](https://doi.org/10.1137/0917003)

  The source of the adaptive inner-tolerance schedule, implemented here as the
  `adaptiveTolerance` switch and its four keys. The forcing term for each
  solve is formed from the ratio of successive nonlinear block residuals, so the
  linear solve is asked for exactly the accuracy the nonlinear iteration can
  still use - loose while the outer residual falls steeply, tightening as it
  flattens; `adaptiveGamma` and `adaptiveAlpha` are the paper's two parameters.
  Its safeguard is implemented alongside the schedule, holding the forcing term
  up to `adaptiveGamma` times the previous term raised to `adaptiveAlpha`
  whenever that quantity exceeds 0.1, so that one unusually good outer iteration
  cannot discard the accuracy the next one needs; it goes quiet of its own
  accord once the forcing terms are small, and `relTolMin` and `relTolMax` clamp
  what survives. The forcing term is defined against the residual of the current
  iterate, which is why the adaptive path also selects PETSc's
  initial-residual-relative convergence test: under the default
  `||b||`-relative test a warm-started loose solve is already converged on entry
  and returns without taking a Krylov step.

- **Henson & Yang (2002)**

  V. E. Henson and U. M. Yang. "BoomerAMG: a parallel algebraic multigrid solver
  and preconditioner". *Applied Numerical Mathematics*, 41(1):155-177, 2002.
  doi:[10.1016/S0168-9274(01)00115-5](https://doi.org/10.1016/S0168-9274(01)00115-5)

  The algebraic multigrid method applied to the pressure Schur complement in the
  default parallel configuration. Replacing the smoothed-aggregation multigrid on
  that block with BoomerAMG was worth 2.7x on the case it was first measured on.
  Its defaults are not size-neutral, though: `strong_threshold 0.25` is a
  two-dimensional value, and retuning it to hypre's recommended 0.5 for three
  dimensions, with HMIS coarsening and truncated `ext+i` interpolation, was in
  turn the single largest step in tuning the solver for large three-dimensional
  meshes (doc/BENCHMARKS.md section 9.1).

- **Falgout & Yang (2002)**

  R. D. Falgout and U. M. Yang. "hypre: a library of high performance
  preconditioners". In *Computational Science - ICCS 2002*, Lecture Notes in
  Computer Science vol. 2331, Springer, pp. 632-641, 2002.
  doi:[10.1007/3-540-47789-6_66](https://doi.org/10.1007/3-540-47789-6_66)

  The library providing BoomerAMG. A PETSc build configured with hypre is a
  prerequisite for the default parallel preconditioner; without it the first
  linear solve fails with an unknown `PCType`. Either rebuild PETSc with hypre
  or override `petscOptions` (see doc/USAGE.md).

- **Balay et al. (PETSc)**

  S. Balay, S. Abhyankar, M. F. Adams, S. Benson, J. Brown, P. Brune,
  K. Buschelman, E. Constantinescu, L. Dalcin, A. Dener, V. Eijkhout,
  J. Faibussowitsch, W. D. Gropp, V. Hapla, T. Isaac, P. Jolivet, D. Karpeev,
  D. Kaushik, M. G. Knepley, F. Kong, S. Kruger, D. A. May, L. C. McInnes,
  R. T. Mills, L. Mitchell, T. Munson, J. E. Roman, K. Rupp, P. Sanan,
  J. Sarich, B. F. Smith, H. Suh, S. Zampini, H. Zhang and J. Zhang.
  *PETSc/TAO Users Manual*, Technical Report ANL-21/39 Revision 3.25, Argonne
  National Laboratory, 2026.
  doi:[10.2172/2998643](https://doi.org/10.2172/2998643) ·
  [petsc.org](https://petsc.org)

  The parallel linear-algebra backend. The solver assembles into `MATMPIAIJ`
  through the COO interface, sets a block size of 4 on the matrix, and solves
  with `KSP`; every preconditioner choice is exposed as a runtime PETSc options
  string. The declared block size supplies the field-split index sets, but it is
  not forwarded to external packages: hypre's systems-AMG options need
  `-pc_hypre_boomeramg_numfunctions 4` passed explicitly, or the library runs at
  one function per node and every block-aware option degenerates silently.

- **Brown, Knepley, May, McInnes & Smith (2012)**

  J. Brown, M. G. Knepley, D. A. May, L. C. McInnes and B. Smith. "Composable
  linear solvers for multiphysics". In *2012 11th International Symposium on
  Parallel and Distributed Computing (ISPDC)*, pp. 55-62, 2012.
  doi:[10.1109/ISPDC.2012.16](https://doi.org/10.1109/ISPDC.2012.16)

  The design of PETSc's `PCFIELDSPLIT`, which lets the velocity rows and the
  pressure rows of the assembled system be preconditioned by different methods
  and composed as a Schur factorisation - the mechanism the default
  configuration relies on.

- **Bnà, Spisso, Olesen & Rossi (2020)**

  S. Bnà, I. Spisso, M. Olesen and G. Rossi. *PETSc4FOAM: a library to plug-in
  PETSc into the OpenFOAM framework*. PRACE white paper WP294, 2020.
  doi:[10.5281/zenodo.3923780](https://doi.org/10.5281/zenodo.3923780) ·
  [develop.openfoam.com/modules/external-solver](https://develop.openfoam.com/modules/external-solver)

  The prior art for coupling PETSc to OpenFOAM: caching of the sparsity pattern,
  the mapping from OpenFOAM's addressing to PETSc's global row indices, and the
  cost accounting for repeated assembly - all of which informed this solver's
  own assembly path, which builds a global-index system directly rather than
  translating a segregated one.

- **Guennebaud, Jacob et al. (Eigen)**

  G. Guennebaud, B. Jacob et al. *Eigen v3*, 2010-.
  [gitlab.com/libeigen/eigen](https://gitlab.com/libeigen/eigen) ·
  [eigen.tuxfamily.org](https://eigen.tuxfamily.org)

  The serial, header-only backend. `SparseLU` provides a direct solve of the
  block system that serves as the exact-arithmetic oracle in validation, and
  `BiCGSTAB` with an incomplete-LU preconditioner provides the iterative serial
  alternative.

- **Demmel, Eisenstat, Gilbert, Li & Liu (1999)**

  J. W. Demmel, S. C. Eisenstat, J. R. Gilbert, X. S. Li and J. W. H. Liu. "A
  supernodal approach to sparse partial pivoting". *SIAM Journal on Matrix
  Analysis and Applications*, 20(3):720-755, 1999.
  doi:[10.1137/S0895479895291765](https://doi.org/10.1137/S0895479895291765)

  The supernodal left-looking factorisation underlying Eigen's `SparseLU`,
  including the fill-reducing ordering and the separation of symbolic from
  numeric factorisation that the solver exploits by reusing the symbolic
  analysis across outer iterations.

---

## 4. OpenFOAM

- **Weller, Tabor, Jasak & Fureby (1998)**

  H. G. Weller, G. Tabor, H. Jasak and C. Fureby. "A tensorial approach to
  computational continuum mechanics using object-oriented techniques".
  *Computers in Physics*, 12(6):620-631, 1998.
  doi:[10.1063/1.168744](https://doi.org/10.1063/1.168744)

  The paper describing the library this solver is written against: the mesh,
  field and `fvMatrix` abstractions whose assembled coefficients the coupled
  system reuses instead of rediscretising the equations from scratch.

- **Greenshields (OpenFOAM v14 User Guide)**

  C. J. Greenshields. *OpenFOAM v14 User Guide*. The OpenFOAM Foundation,
  London, 2026.
  [doc.cfd.direct/openfoam/user-guide-v14](https://doc.cfd.direct/openfoam/user-guide-v14/)

  The reference for case structure, dictionary syntax and the `fvSchemes` and
  `fvSolution` entries the solver reads, and the source of the conventions the
  `COUPLED` dictionary follows.

- **The OpenFOAM Foundation**

  OpenFOAM, distributed by The OpenFOAM Foundation.
  [openfoam.org](https://openfoam.org)

  The distribution this solver targets. It builds against OpenFOAM 14 from the
  Foundation line (openfoam.org) and uses its `momentumTransport` library for
  turbulence; it is not tested against other distributions.

- **Trademark note**

  OPENFOAM® is a registered trade mark of OpenCFD Limited, producer and
  distributor of the OpenFOAM software via www.openfoam.com. This repository is
  an independent work: it is not approved or endorsed by OpenCFD Limited, nor by
  The OpenFOAM Foundation, and the trade mark is used here only to identify the
  software the solver is compatible with.

---

## 5. Turbulence models

Turbulence is not part of the coupled system: the models below are solved
segregated and lagged one outer iteration behind the coupled momentum-continuity
solve, using OpenFOAM's `momentumTransport` library. The references are those the
library's own implementations cite.

- **Launder & Spalding (1974)**

  B. E. Launder and D. B. Spalding. "The numerical computation of turbulent
  flows". *Computer Methods in Applied Mechanics and Engineering*, 3(2):269-289,
  1974.
  doi:[10.1016/0045-7825(74)90029-2](https://doi.org/10.1016/0045-7825(74)90029-2)

  The standard k-epsilon model as implemented by OpenFOAM's `kEpsilon`, the RANS
  model exercised in this solver's turbulent validation cases.

- **Menter (1994)**

  F. R. Menter. "Two-equation eddy-viscosity turbulence models for engineering
  applications". *AIAA Journal*, 32(8):1598-1605, 1994.
  doi:[10.2514/3.12149](https://doi.org/10.2514/3.12149)

  The original k-omega SST model: the blending functions and the shear-stress
  limiter that define the second RANS model available to the solver.

- **Menter & Esch (2001)**

  F. R. Menter and T. Esch. "Elements of industrial heat transfer prediction".
  16th Brazilian Congress of Mechanical Engineering (COBEM), Uberlândia, Brazil,
  2001.

  The formulation OpenFOAM's `kOmegaSST` implementation states it follows, in
  particular the production-limiter form.

- **Menter, Kuntz & Langtry (2003)**

  F. R. Menter, M. Kuntz and R. Langtry. "Ten years of industrial experience with
  the SST turbulence model". In K. Hanjalić, Y. Nagano and M. Tummers (eds.),
  *Turbulence, Heat and Mass Transfer 4*, Begell House, pp. 625-632, 2003.

  The updated model coefficients used by OpenFOAM's `kOmegaSST`.

- **Hellsten (1998)**

  A. Hellsten. "Some improvements in Menter's k-omega SST turbulence model".
  29th AIAA Fluid Dynamics Conference, Albuquerque, NM, AIAA paper 1998-2554,
  1998.
  doi:[10.2514/6.1998-2554](https://doi.org/10.2514/6.1998-2554)

  The F3 rough-wall correction carried by the same OpenFOAM implementation;
  listed for completeness, since selecting it changes the eddy viscosity the
  coupled momentum operator sees.

---

## 6. Further reading

- **Ferziger, Perić & Street (2020)**

  J. H. Ferziger, M. Perić and R. L. Street. *Computational Methods for Fluid
  Dynamics*, 4th edition. Springer, Cham, 2020.
  doi:[10.1007/978-3-319-99693-6](https://doi.org/10.1007/978-3-319-99693-6)

  A broader treatment of pressure-velocity coupling, including the relation
  between coupled and segregated formulations and the effect of grid quality on
  both.

- **Versteeg & Malalasekera (2007)**

  H. K. Versteeg and W. Malalasekera. *An Introduction to Computational Fluid
  Dynamics: The Finite Volume Method*, 2nd edition. Pearson Education, Harlow,
  2007. ISBN 978-0-13-127498-3.

  An accessible route into the same finite-volume material for readers meeting
  SIMPLE, upwinding and boundary-condition treatment for the first time.

- **Saad (2003)**

  Y. Saad. *Iterative Methods for Sparse Linear Systems*, 2nd edition. SIAM,
  Philadelphia, 2003.
  doi:[10.1137/1.9780898718003](https://doi.org/10.1137/1.9780898718003)

  Background for the Krylov methods and incomplete factorisations named in the
  solver's PETSc and Eigen options - GMRES and its flexible variant, BiCGSTAB,
  ILU and block Jacobi.

- **Notay (2010)**

  Y. Notay. "An aggregation-based algebraic multigrid method". *Electronic
  Transactions on Numerical Analysis*, 37:123-146, 2010.
  [etna.ricam.oeaw.ac.at/vol.37.2010/pp123-146.dir/pp123-146.pdf](https://etna.ricam.oeaw.ac.at/vol.37.2010/pp123-146.dir/pp123-146.pdf)

  Pairwise aggregation with a K-cycle: the cheap-setup multigrid design that
  matters when the operator changes every outer iteration, as it does here, and
  the starting point for anyone implementing a purpose-built block-AMG for this
  matrix. That is now the only open route to one, the same architecture built
  from generic library components having been measured and rejected on cost
  (doc/BENCHMARKS.md section 9.3). Setup cost is still the quantity that
  decides the design: on the largest case the tuned three-dimensional hierarchy
  is cheap enough per cycle that rebuilding it only every eighth outer iteration
  is the faster choice, which is the same economics seen from the other side.

- **Jasak (1996)**

  H. Jasak. *Error Analysis and Estimation for the Finite Volume Method with
  Applications to Fluid Flows*. PhD thesis, Imperial College London, 1996.
  [hdl.handle.net/10044/1/8335](https://hdl.handle.net/10044/1/8335)

  The origin of much of OpenFOAM's discretisation practice, including the
  non-orthogonal correction and boundary treatment whose coefficients the coupled
  assembly inherits from `fvMatrix`.
