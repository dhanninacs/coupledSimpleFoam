/*---------------------------------------------------------------------------*\
  coupledSimpleFoam - Copyright (C) 2026 dhanninacs - GPL-3.0-or-later
\*---------------------------------------------------------------------------*/

#include "conjugateEnergy.H"

#include <algorithm>
#include <cmath>
#include <sstream>
#include <string>

#include "fvm.H"
#include "fvc.H"
#include "fvMatrices.H"
#include "IOdictionary.H"
#include "Residuals.H"
#include "Pstream.H"

#include <Eigen/Sparse>
#include <Eigen/IterativeLinearSolvers>

#ifdef HAVE_AMGX
#include "amgxCommon.H"
#endif

namespace
{
#ifdef HAVE_AMGX
    struct amgxState
    {
        AMGX_matrix_handle A = nullptr;
        AMGX_vector_handle b = nullptr;
        AMGX_vector_handle x = nullptr;
        bool uploaded = false;
    };

    const char* const conjugateConfig =
        "config_version=2, exception_handling=1, "
        "solver(main)=PBICGSTAB, "
        "main:convergence=RELATIVE_INI_CORE, main:norm=L2, "
        "main:monitor_residual=1, main:print_solve_stats=0, "
        "main:preconditioner(amg)=AMG, "
        "amg:algorithm=AGGREGATION, amg:selector=SIZE_2, "
        "amg:smoother(sm)=MULTICOLOR_DILU, sm:relaxation_factor=1.0, "
        "amg:presweeps=1, amg:postsweeps=1, amg:max_iters=1, "
        "amg:cycle=V, amg:max_levels=50, "
        "amg:coarse_solver=DENSE_LU_SOLVER, amg:dense_lu_num_rows=64, "
        "amg:min_coarse_rows=32";
#endif

    Foam::scalar readKappa(const Foam::fvMesh& solid)
    {
        Foam::IOdictionary props
        (
            Foam::IOobject
            (
                "physicalProperties",
                solid.time().constant(),
                solid,
                Foam::IOobject::MUST_READ,
                Foam::IOobject::NO_WRITE,
                false
            )
        );
        if (props.found("kappa"))
        {
            return props.lookup<Foam::scalar>("kappa");
        }
        return props.subDict("mixture").subDict("transport").lookup<Foam::scalar>("kappa");
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::conjugateEnergy::conjugateEnergy
(
    fvMesh& fluid,
    volScalarField& Tf,
    const dictionary& coupledDict
)
:
    fluid_(fluid),
    Tf_(Tf),
    rhoCp_(coupledDict.lookup<scalar>("rhoCp")),
    solverName_(coupledDict.lookupOrDefault<word>("conjugateSolver", "Eigen")),
    tolerance_(coupledDict.lookupOrDefault<scalar>("conjugateTolerance", 1e-10)),
    maxIter_(coupledDict.lookupOrDefault<label>("conjugateMaxIter", 1000)),
    n_(0),
    patternBuilt_(false),
    amgx_(nullptr)
{
    if (Pstream::parRun())
    {
        FatalErrorInFunction
            << "the conjugate energy solve is serial (one rank)"
            << exit(FatalError);
    }
    if (solverName_ != "AmgX" && solverName_ != "Eigen")
    {
        FatalErrorInFunction
            << "conjugateSolver is AmgX or Eigen (got " << solverName_ << ")"
            << exit(FatalError);
    }
#ifndef HAVE_AMGX
    if (solverName_ == "AmgX")
    {
        FatalErrorInFunction
            << "conjugateSolver AmgX requested but this build has no AmgX"
            << exit(FatalError);
    }
#endif

    const wordList names(coupledDict.lookup("solidRegions"));
    solids_.setSize(names.size());
    Ts_.setSize(names.size());
    kappa_.setSize(names.size());

    const Time& runTime = fluid_.time();
    forAll(names, s)
    {
        solids_.set
        (
            s,
            new fvMesh
            (
                IOobject(names[s], runTime.name(), runTime, IOobject::MUST_READ)
            )
        );
        Ts_.set
        (
            s,
            new volScalarField
            (
                IOobject
                (
                    "T",
                    runTime.name(),
                    solids_[s],
                    IOobject::MUST_READ,
                    IOobject::AUTO_WRITE
                ),
                solids_[s]
            )
        );
        kappa_[s] = readKappa(solids_[s]);
        Info<< "Conjugate region " << names[s] << ": " << solids_[s].nCells()
            << " cells, kappa = " << kappa_[s] << endl;
    }

    offset_.setSize(names.size() + 1);
    offset_[0] = 0;
    n_ = fluid_.nCells();
    forAll(solids_, s)
    {
        offset_[s + 1] = n_;
        n_ += solids_[s].nCells();
    }

    matchInterfaces();

    Info<< "Conjugate energy: " << n_ << " cells in one system, "
        << ifaces_.size() << " interface(s), rhoCp " << rhoCp_
        << ", solver " << solverName_ << nl << endl;
}


Foam::conjugateEnergy::~conjugateEnergy()
{
#ifdef HAVE_AMGX
    if (amgx_)
    {
        amgxState* st = static_cast<amgxState*>(amgx_);
        if (st->x) AMGX_vector_destroy(st->x);
        if (st->b) AMGX_vector_destroy(st->b);
        if (st->A) AMGX_matrix_destroy(st->A);
        delete st;
        amgxCommon::release();
    }
#endif
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::conjugateEnergy::matchInterfaces()
{
    const word fluidName = fluid_.name();
    forAll(solids_, s)
    {
        const word solidName = solids_[s].name();
        auto patchIndex = [](const fvMesh& mesh, const word& name)
        {
            forAll(mesh.boundary(), pi)
            {
                if (mesh.boundary()[pi].name() == name) return pi;
            }
            return label(-1);
        };
        const label fp = patchIndex(fluid_, fluidName + "_to_" + solidName);
        const label sp = patchIndex(solids_[s], solidName + "_to_" + fluidName);
        if (fp < 0 || sp < 0)
        {
            continue;  // this solid does not touch the fluid
        }

        const vectorField& cf = fluid_.boundary()[fp].Cf();
        const vectorField& cs = solids_[s].boundary()[sp].Cf();
        if (cf.size() != cs.size())
        {
            FatalErrorInFunction
                << "interface " << fluidName << "_to_" << solidName << " has "
                << cf.size() << " faces but its other side has " << cs.size()
                << "; the conjugate solve needs a conformal interface"
                << exit(FatalError);
        }

        // Match by face centre: sort both sides lexicographically, pair in
        // order, and check every pair.
        auto order = [](const vectorField& c)
        {
            std::vector<label> idx(c.size());
            for (label i = 0; i < c.size(); i++) idx[i] = i;
            std::sort
            (
                idx.begin(), idx.end(),
                [&](label a, label b)
                {
                    const scalar tol = 1e-9;
                    if (std::abs(c[a].x() - c[b].x()) > tol) return c[a].x() < c[b].x();
                    if (std::abs(c[a].y() - c[b].y()) > tol) return c[a].y() < c[b].y();
                    return c[a].z() < c[b].z();
                }
            );
            return idx;
        };
        const std::vector<label> of = order(cf);
        const std::vector<label> os = order(cs);

        const scalar span = mag(max(cf) - min(cf));
        interface ifc;
        ifc.solid = s;
        ifc.fluidPatch = fp;
        ifc.solidPatch = sp;
        ifc.fluidFace.setSize(cf.size());
        ifc.solidFace.setSize(cs.size());
        for (label i = 0; i < cf.size(); i++)
        {
            ifc.fluidFace[i] = of[i];
            ifc.solidFace[i] = os[i];
            if (mag(cf[of[i]] - cs[os[i]]) > 1e-6*max(span, small))
            {
                FatalErrorInFunction
                    << "interface faces do not match at " << cf[of[i]]
                    << " / " << cs[os[i]]
                    << "; the conjugate solve needs a conformal interface"
                    << exit(FatalError);
            }
        }
        Info<< "Conjugate interface " << fluidName << " <-> " << solidName
            << ": " << cf.size() << " matched faces" << endl;
        ifaces_.append(ifc);
    }
}


void Foam::conjugateEnergy::buildPattern()
{
    // The (row, col) of every contribution, in exactly the order assemble()
    // produces the values.
    std::vector<int> rows;
    std::vector<int> cols;

    auto region = [&](const fvMesh& mesh, label off)
    {
        const label nc = mesh.nCells();
        for (label i = 0; i < nc; i++)
        {
            rows.push_back(off + i); cols.push_back(off + i);
        }
        const labelUList& l = mesh.lduAddr().lowerAddr();
        const labelUList& u = mesh.lduAddr().upperAddr();
        forAll(l, f)
        {
            rows.push_back(off + l[f]); cols.push_back(off + u[f]);   // upper
        }
        forAll(l, f)
        {
            rows.push_back(off + u[f]); cols.push_back(off + l[f]);   // lower
        }
    };
    region(fluid_, 0);
    forAll(solids_, s)
    {
        region(solids_[s], offset_[s + 1]);
    }
    for (const interface& ifc : ifaces_)
    {
        const labelUList& fcF = fluid_.boundary()[ifc.fluidPatch].faceCells();
        const labelUList& fcS = solids_[ifc.solid].boundary()[ifc.solidPatch].faceCells();
        const label so = offset_[ifc.solid + 1];
        forAll(ifc.fluidFace, i)
        {
            const int a = fcF[ifc.fluidFace[i]];
            const int b = so + fcS[ifc.solidFace[i]];
            rows.push_back(a); cols.push_back(a);
            rows.push_back(a); cols.push_back(b);
            rows.push_back(b); cols.push_back(b);
            rows.push_back(b); cols.push_back(a);
        }
    }

    // CSR with sorted, unique columns per row, and each contribution's slot.
    const size_t nc = rows.size();
    std::vector<std::vector<int>> perRow(n_);
    for (size_t k = 0; k < nc; k++)
    {
        perRow[rows[k]].push_back(cols[k]);
    }
    rowPtr_.assign(n_ + 1, 0);
    for (label i = 0; i < n_; i++)
    {
        std::vector<int>& c = perRow[i];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        rowPtr_[i + 1] = rowPtr_[i] + int(c.size());
    }
    colIdx_.resize(rowPtr_[n_]);
    for (label i = 0; i < n_; i++)
    {
        std::copy(perRow[i].begin(), perRow[i].end(), colIdx_.begin() + rowPtr_[i]);
    }
    slot_.resize(nc);
    for (size_t k = 0; k < nc; k++)
    {
        const int r = rows[k];
        const int* first = colIdx_.data() + rowPtr_[r];
        const int* last = colIdx_.data() + rowPtr_[r + 1];
        slot_[k] = int(std::lower_bound(first, last, cols[k]) - colIdx_.data());
    }
    values_.assign(colIdx_.size(), 0.0);
    rhs_.assign(n_, 0.0);
    x_.assign(n_, 0.0);
    patternBuilt_ = true;
}


void Foam::conjugateEnergy::assemble
(
    const surfaceScalarField& phi,
    const volScalarField& alphaEff,
    std::vector<double>& contrib
)
{
    contrib.clear();
    std::fill(rhs_.begin(), rhs_.end(), 0.0);

    // One region's matrix: diagonal (with the boundary coefficients of every
    // non-interface patch, as fvMatrix::solve adds them), upper, lower, and
    // its source into the right-hand side.
    auto region = [&]
    (
        fvScalarMatrix& M,
        const fvMesh& mesh,
        label off,
        const labelHashSet& skip
    )
    {
        scalarField diag(M.diag());
        scalarField source(M.source());
        forAll(mesh.boundary(), pi)
        {
            if (skip.found(pi) || mesh.boundary()[pi].coupled())
            {
                continue;
            }
            const labelUList& fc = mesh.boundary()[pi].faceCells();
            const scalarField& ic = M.internalCoeffs()[pi];
            const scalarField& bc = M.boundaryCoeffs()[pi];
            forAll(fc, i)
            {
                diag[fc[i]] += ic[i];
                source[fc[i]] += bc[i];
            }
        }
        for (label i = 0; i < mesh.nCells(); i++)
        {
            contrib.push_back(diag[i]);
            rhs_[off + i] = source[i];
        }
        const scalarField& upper = M.upper();
        const scalarField& lower = M.hasLower() ? M.lower() : M.upper();
        forAll(upper, f) contrib.push_back(upper[f]);
        forAll(lower, f) contrib.push_back(lower[f]);
    };

    // Fluid: the same equation as the segregated path, in W/K.
    {
        fvScalarMatrix TEqn(fvm::div(phi, Tf_) - fvm::laplacian(alphaEff, Tf_));
        TEqn *= dimensionedScalar("rhoCp", dimless, rhoCp_);
        labelHashSet skip;
        for (const interface& ifc : ifaces_) skip.insert(ifc.fluidPatch);
        region(TEqn, fluid_, 0, skip);
    }
    forAll(solids_, s)
    {
        volScalarField& T = Ts_[s];
        fvScalarMatrix TEqn
        (
            -fvm::laplacian(dimensionedScalar("kappa", dimless, kappa_[s]), T)
        );
        labelHashSet skip;
        for (const interface& ifc : ifaces_)
        {
            if (ifc.solid == s) skip.insert(ifc.solidPatch);
        }
        region(TEqn, solids_[s], offset_[s + 1], skip);
    }

    // Interfaces: series conductance between the two cells of each pair.
    for (const interface& ifc : ifaces_)
    {
        const fvPatch& pF = fluid_.boundary()[ifc.fluidPatch];
        const fvPatch& pS = solids_[ifc.solid].boundary()[ifc.solidPatch];
        const scalarField& dF = pF.deltaCoeffs();
        const scalarField& dS = pS.deltaCoeffs();
        const scalarField& aF = alphaEff.boundaryField()[ifc.fluidPatch];
        const scalarField magSf(pF.magSf());
        forAll(ifc.fluidFace, i)
        {
            const label f = ifc.fluidFace[i];
            const label g = ifc.solidFace[i];
            const scalar hF = rhoCp_*aF[f]*dF[f];
            const scalar hS = kappa_[ifc.solid]*dS[g];
            const scalar G = magSf[f]/(1.0/hF + 1.0/hS);
            contrib.push_back(G);
            contrib.push_back(-G);
            contrib.push_back(G);
            contrib.push_back(-G);
        }
    }
}


void Foam::conjugateEnergy::residual(scalar& res, scalar& normFactor) const
{
    // OpenFOAM's normalised residual (lduMatrix::solver::normFactor) of the
    // combined system at x_.
    scalar xRef = 0.0;
    for (label i = 0; i < n_; i++) xRef += x_[i];
    xRef /= max(scalar(n_), 1.0);
    scalar sumR = 0.0;
    scalar sumN = 0.0;
    #pragma omp parallel for reduction(+:sumR, sumN) schedule(static)
    for (label i = 0; i < n_; i++)
    {
        scalar ax = 0.0;
        scalar rowSum = 0.0;
        for (int q = rowPtr_[i]; q < rowPtr_[i + 1]; q++)
        {
            ax += values_[q]*x_[colIdx_[q]];
            rowSum += values_[q];
        }
        const scalar axRef = rowSum*xRef;
        sumR += std::abs(rhs_[i] - ax);
        sumN += std::abs(ax - axRef) + std::abs(rhs_[i] - axRef);
    }
    normFactor = sumN + small;
    res = sumR/normFactor;
}


Foam::label Foam::conjugateEnergy::solveSystem(scalar initialResidual, scalar normFactor)
{
    const scalar target = tolerance_;
    if (solverName_ == "Eigen")
    {
        typedef Eigen::SparseMatrix<double, Eigen::RowMajor> SpMat;
        Eigen::Map<const SpMat> A
        (
            n_, n_, int(colIdx_.size()), rowPtr_.data(), colIdx_.data(), values_.data()
        );
        Eigen::Map<const Eigen::VectorXd> b(rhs_.data(), n_);
        Eigen::Map<Eigen::VectorXd> x(x_.data(), n_);
        Eigen::BiCGSTAB<SpMat, Eigen::IncompleteLUT<double>> solver;
        solver.setMaxIterations(maxIter_);
        // Eigen's tolerance is ||r||_2/||b||_2; aim at the OpenFOAM target
        // with margin, and let the caller's host check decide.
        const scalar bNorm = max(b.norm(), small);
        solver.setTolerance(std::min(0.1, 0.1*target*normFactor/(bNorm*std::sqrt(scalar(n_)))));
        // Eigen's iterative solvers keep a reference to the matrix given to
        // compute(): it must outlive solveWithGuess, so it is a named local
        // (a temporary here was destroyed before the solve, FoamWin #2).
        const SpMat Acopy(A);
        solver.compute(Acopy);
        Eigen::VectorXd guess = x;
        x = solver.solveWithGuess(b, guess);
        return label(solver.iterations());
    }

#ifdef HAVE_AMGX
    if (!amgx_)
    {
        amgxCommon::acquire();
        amgxState* st = new amgxState;
        AMGX_resources_handle rsrc = amgxCommon::resources();
        amgxCommon::check(AMGX_matrix_create(&st->A, rsrc, AMGX_mode_dDDI), "matrix_create");
        amgxCommon::check(AMGX_vector_create(&st->b, rsrc, AMGX_mode_dDDI), "vector_create");
        amgxCommon::check(AMGX_vector_create(&st->x, rsrc, AMGX_mode_dDDI), "vector_create");
        amgx_ = st;
    }
    amgxState* st = static_cast<amgxState*>(amgx_);
    const int nnz = int(colIdx_.size());
    if (!st->uploaded)
    {
        amgxCommon::check
        (
            AMGX_matrix_upload_all
            (
                st->A, int(n_), nnz, 1, 1, rowPtr_.data(), colIdx_.data(),
                values_.data(), nullptr
            ),
            "matrix_upload_all"
        );
        st->uploaded = true;
    }
    else
    {
        amgxCommon::check
        (
            AMGX_matrix_replace_coefficients(st->A, int(n_), nnz, values_.data(), nullptr),
            "matrix_replace_coefficients"
        );
    }
    amgxCommon::check(AMGX_vector_upload(st->b, int(n_), 1, rhs_.data()), "vector_upload rhs");
    amgxCommon::check(AMGX_vector_upload(st->x, int(n_), 1, x_.data()), "vector_upload x");

    const scalar amgxTol =
        std::min(scalar(0.1), 0.5*target/std::max(initialResidual, scalar(vSmall)));
    std::ostringstream full;
    full.precision(6);
    full << conjugateConfig << ", main:tolerance=" << amgxTol
         << ", main:max_iters=" << maxIter_;
    AMGX_config_handle cfg;
    amgxCommon::check(AMGX_config_create(&cfg, full.str().c_str()), "config_create");
    AMGX_solver_handle slv;
    amgxCommon::check
    (
        AMGX_solver_create(&slv, amgxCommon::resources(), AMGX_mode_dDDI, cfg),
        "solver_create"
    );
    amgxCommon::check(AMGX_solver_setup(slv, st->A), "solver_setup");
    amgxCommon::check(AMGX_solver_solve(slv, st->b, st->x), "solver_solve");
    int its = 0;
    AMGX_solver_get_iterations_number(slv, &its);
    amgxCommon::check(AMGX_vector_download(st->x, x_.data()), "vector_download");
    AMGX_solver_destroy(slv);
    AMGX_config_destroy(cfg);
    return label(its);
#else
    return 0;
#endif
}


void Foam::conjugateEnergy::solve
(
    const surfaceScalarField& phi,
    const volScalarField& alphaEff
)
{
    if (!patternBuilt_)
    {
        buildPattern();
    }

    std::vector<double> contrib;
    contrib.reserve(slot_.size());
    assemble(phi, alphaEff, contrib);
    if (contrib.size() != slot_.size())
    {
        FatalErrorInFunction
            << "conjugate assembly produced " << contrib.size()
            << " contributions against a pattern of " << slot_.size()
            << exit(FatalError);
    }
    std::fill(values_.begin(), values_.end(), 0.0);
    for (size_t k = 0; k < contrib.size(); k++)
    {
        values_[slot_[k]] += contrib[k];
    }

    // Current temperatures as the initial guess.
    for (label i = 0; i < fluid_.nCells(); i++) x_[i] = Tf_[i];
    forAll(solids_, s)
    {
        const label off = offset_[s + 1];
        for (label i = 0; i < solids_[s].nCells(); i++) x_[off + i] = Ts_[s][i];
    }

    solverPerformance perf(solverName_ + "Conjugate", Tf_.name());
    scalar res0 = 0.0;
    scalar normFactor = 1.0;
    residual(res0, normFactor);
    perf.initialResidual() = res0;
    perf.finalResidual() = res0;
    if (res0 > tolerance_)
    {
        perf.nIterations() = solveSystem(res0, normFactor);
        scalar res1 = 0.0;
        scalar nf1 = 1.0;
        residual(res1, nf1);
        perf.finalResidual() = res1;
        perf.checkConvergence(tolerance_, 0);
    }
    perf.print(Info);
    Residuals<scalar>::append(fluid_, perf);

    // Distribute, then the interface temperature onto both interface patches.
    for (label i = 0; i < fluid_.nCells(); i++) Tf_.primitiveFieldRef()[i] = x_[i];
    forAll(solids_, s)
    {
        const label off = offset_[s + 1];
        scalarField& Ti = Ts_[s].primitiveFieldRef();
        for (label i = 0; i < solids_[s].nCells(); i++) Ti[i] = x_[off + i];
    }
    for (const interface& ifc : ifaces_)
    {
        const fvPatch& pF = fluid_.boundary()[ifc.fluidPatch];
        const fvPatch& pS = solids_[ifc.solid].boundary()[ifc.solidPatch];
        const labelUList& fcF = pF.faceCells();
        const labelUList& fcS = pS.faceCells();
        const scalarField& dF = pF.deltaCoeffs();
        const scalarField& dS = pS.deltaCoeffs();
        const scalarField& aF = alphaEff.boundaryField()[ifc.fluidPatch];
        scalarField tF(pF.size(), 0.0);
        scalarField tS(pS.size(), 0.0);
        forAll(ifc.fluidFace, i)
        {
            const label f = ifc.fluidFace[i];
            const label g = ifc.solidFace[i];
            const scalar hF = rhoCp_*aF[f]*dF[f];
            const scalar hS = kappa_[ifc.solid]*dS[g];
            const scalar Ta = Tf_[fcF[f]];
            const scalar Tb = Ts_[ifc.solid][fcS[g]];
            const scalar Tw = (hF*Ta + hS*Tb)/(hF + hS);
            tF[f] = Tw;
            tS[g] = Tw;
        }
        Tf_.boundaryFieldRef()[ifc.fluidPatch] == tF;
        Ts_[ifc.solid].boundaryFieldRef()[ifc.solidPatch] == tS;
    }
    Tf_.correctBoundaryConditions();
    forAll(Ts_, s)
    {
        Ts_[s].correctBoundaryConditions();
    }
}


// ************************************************************************* //
