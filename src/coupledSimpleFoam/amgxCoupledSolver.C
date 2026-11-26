/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | coupledSimpleFoam
   \\    /   O peration     | Fully coupled steady incompressible p-U solver
    \\  /    A nd           | for OpenFOAM
     \\/     M anipulation  |
-------------------------------------------------------------------------------
Copyright
    Copyright (C) 2026 dhanninacs

License
    This file is part of coupledSimpleFoam, a solver application for OpenFOAM.

    coupledSimpleFoam is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    coupledSimpleFoam is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY
    or FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License along
    with coupledSimpleFoam.  If not, see <http://www.gnu.org/licenses/>.

Description
    AmgX (serial GPU) backend for the coupled p-U system.  The whole
    translation unit compiles to nothing when the build did not find AmgX.

\*---------------------------------------------------------------------------*/

#include "amgxCoupledSolver.H"

#ifdef HAVE_AMGX

#include <algorithm>
#include <chrono>
#include <climits>
#include <sstream>

#include <amgx_c.h>

#include "amgxCommon.H"

#include "IOstreams.H"
#include "Pstream.H"
#include "error.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace
{
    using Foam::amgxCommon::check;

    //- Block-AMG default: the configuration amgxCoupled4Foam reported for
    //  coupled block systems (BiCGStab outer, aggregation AMG, block DILU).
    const char* const defaultConfig =
        "config_version=2, exception_handling=1, "
        "solver(main)=PBICGSTAB, "
        "main:convergence=RELATIVE_INI_CORE, main:norm=L2, "
        "main:monitor_residual=1, main:print_solve_stats=0, "
        "main:obtain_timings=0, "
        "main:preconditioner(amg)=AMG, "
        "amg:algorithm=AGGREGATION, amg:selector=SIZE_2, "
        "amg:smoother(sm)=MULTICOLOR_DILU, sm:relaxation_factor=1.0, "
        "amg:presweeps=1, amg:postsweeps=1, amg:max_iters=1, "
        "amg:cycle=V, amg:max_levels=50, "
        "amg:coarse_solver=DENSE_LU_SOLVER, amg:dense_lu_num_rows=64, "
        "amg:min_coarse_rows=32, amg:print_grid_stats=0";
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::amgxCoupledSolver::amgxCoupledSolver(const dictionary& dict)
:
    name_("AmgX"),
    relTolerance_(dict.lookupOrDefault<scalar>("relTol", 1e-6)),
    maxIter_(dict.lookupOrDefault<label>("maxLinearIter", 500)),
    pcReuseInterval_(dict.lookupOrDefault<label>("pcReuseInterval", 1)),
    reuse_(dict.lookupOrDefault<word>("amgxReuse", "resetup")),
    mixed_(dict.lookupOrDefault<word>("amgxPrecision", "double") == "mixed"),
    verbose_(dict.lookupOrDefault<bool>("amgxVerbose", false)),
    config_(dict.lookupOrDefault<string>("amgxConfig", string(defaultConfig))),
    rsrc_(nullptr),
    cfg_(nullptr),
    solverCfg_(nullptr),
    mat_(nullptr),
    rhs_(nullptr),
    sol_(nullptr),
    solver_(nullptr),
    nBlockRows_(0),
    nCoo_(-1),
    nSolves_(0),
    uploaded_(false)
{
    if (Pstream::parRun())
    {
        FatalErrorInFunction
            << "linearSolver AmgX is serial (one rank, one GPU); run without"
            << " -parallel or use linearSolver PETSc." << exit(FatalError);
    }
    if (pcReuseInterval_ < 1)
    {
        FatalErrorInFunction
            << "pcReuseInterval must be >= 1 (got " << pcReuseInterval_ << ")"
            << exit(FatalError);
    }
    if (reuse_ != "none" && reuse_ != "resetup")
    {
        FatalErrorInFunction
            << "amgxReuse is none or resetup (got " << reuse_ << ")"
            << exit(FatalError);
    }

    amgxCommon::acquire();
    amgxCommon::setVerbose(verbose_);

    AMGX_config_handle cfg;
    check(AMGX_config_create(&cfg, config_.c_str()), "config_create");
    cfg_ = static_cast<void*>(cfg);

    // The process's one resources object (amgxCommon), not one of our own.
    AMGX_resources_handle rsrc = amgxCommon::resources();
    rsrc_ = static_cast<void*>(rsrc);

    const AMGX_Mode mode = mixed_ ? AMGX_mode_dDFI : AMGX_mode_dDDI;

    AMGX_matrix_handle A;
    AMGX_vector_handle b;
    AMGX_vector_handle s;
    check(AMGX_matrix_create(&A, rsrc, mode), "matrix_create");
    check(AMGX_vector_create(&b, rsrc, mode), "vector_create");
    check(AMGX_vector_create(&s, rsrc, mode), "vector_create");
    mat_ = static_cast<void*>(A);
    rhs_ = static_cast<void*>(b);
    sol_ = static_cast<void*>(s);

    Info<< "AmgX backend: " << (mixed_ ? "mixed (dDFI)" : "double (dDDI)")
        << " precision, preconditioner rebuilt every " << pcReuseInterval_
        << " solve(s), reuse " << reuse_ << endl;
}


Foam::amgxCoupledSolver::~amgxCoupledSolver()
{
    if (solver_)
    {
        AMGX_solver_destroy(static_cast<AMGX_solver_handle>(solver_));
    }
    if (sol_) AMGX_vector_destroy(static_cast<AMGX_vector_handle>(sol_));
    if (rhs_) AMGX_vector_destroy(static_cast<AMGX_vector_handle>(rhs_));
    if (mat_) AMGX_matrix_destroy(static_cast<AMGX_matrix_handle>(mat_));
    if (solverCfg_) AMGX_config_destroy(static_cast<AMGX_config_handle>(solverCfg_));
    if (cfg_) AMGX_config_destroy(static_cast<AMGX_config_handle>(cfg_));
    amgxCommon::release();
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

void Foam::amgxCoupledSolver::buildPattern(const coupledLinearSystem& sys)
{
    const label bs = sys.blockSize();
    if (bs != 4 || sys.nLocalRows() % bs != 0)
    {
        FatalErrorInFunction
            << "expected an interleaved block-4 system, got block size " << bs
            << " and " << sys.nLocalRows() << " rows" << exit(FatalError);
    }

    nBlockRows_ = sys.nLocalRows()/bs;
    nCoo_ = sys.nEntries();

    // Block columns per block row, then sorted and de-duplicated; the
    // diagonal block is always present (the DILU smoother inverts it).
    std::vector<std::vector<int>> cols(nBlockRows_);
    for (label br = 0; br < nBlockRows_; br++)
    {
        cols[br].push_back(static_cast<int>(br));
    }
    for (label k = 0; k < nCoo_; k++)
    {
        cols[sys.row(k)/bs].push_back(static_cast<int>(sys.col(k)/bs));
    }

    rowPtr_.assign(nBlockRows_ + 1, 0);
    for (label br = 0; br < nBlockRows_; br++)
    {
        std::vector<int>& c = cols[br];
        std::sort(c.begin(), c.end());
        c.erase(std::unique(c.begin(), c.end()), c.end());
        rowPtr_[br + 1] = rowPtr_[br] + static_cast<int>(c.size());
    }

    const int64_t nnzb = rowPtr_[nBlockRows_];
    if (nnzb*bs*bs > int64_t(INT_MAX))
    {
        FatalErrorInFunction
            << nnzb << " blocks exceed AmgX's 32-bit value indexing"
            << exit(FatalError);
    }

    colIdx_.resize(nnzb);
    for (label br = 0; br < nBlockRows_; br++)
    {
        std::copy(cols[br].begin(), cols[br].end(), colIdx_.begin() + rowPtr_[br]);
        std::vector<int>().swap(cols[br]);
    }

    slot_.resize(nCoo_);
    for (label k = 0; k < nCoo_; k++)
    {
        const label r = sys.row(k);
        const label c = sys.col(k);
        const int br = static_cast<int>(r/bs);
        const int bc = static_cast<int>(c/bs);
        const int* first = colIdx_.data() + rowPtr_[br];
        const int* last = colIdx_.data() + rowPtr_[br + 1];
        const int64_t b = std::lower_bound(first, last, bc) - colIdx_.data();
        // Row-major within the block: entry (r%bs, c%bs).
        slot_[k] = b*bs*bs + (r % bs)*bs + (c % bs);
    }

    blockRowFirst_.assign(nBlockRows_ + 1, 0);
    for (label k = 0; k < nCoo_; k++)
    {
        blockRowFirst_[sys.row(k)/bs + 1]++;
    }
    for (label br = 0; br < nBlockRows_; br++)
    {
        blockRowFirst_[br + 1] += blockRowFirst_[br];
    }
    byBlockRow_.resize(nCoo_);
    {
        std::vector<int64_t> fill(blockRowFirst_.begin(), blockRowFirst_.end() - 1);
        for (label k = 0; k < nCoo_; k++)
        {
            byBlockRow_[fill[sys.row(k)/bs]++] = static_cast<int>(k);
        }
    }

    valuesD_.assign(nnzb*bs*bs, 0.0);
    if (mixed_)
    {
        valuesF_.assign(nnzb*bs*bs, 0.0f);
    }

    Info<< "AmgX block system: " << nBlockRows_ << " block rows, " << nnzb
        << " 4x4 blocks from " << nCoo_ << " triplets" << endl;
}


void Foam::amgxCoupledSolver::fillValues(const coupledLinearSystem& sys)
{
    // Parallel over block rows: each thread zeroes and fills only the blocks
    // of its own rows, so there are no write conflicts.
    const scalar* v = sys.valData();
    const int bb = 16;
    #pragma omp parallel for schedule(static)
    for (label br = 0; br < nBlockRows_; br++)
    {
        const int64_t first = int64_t(rowPtr_[br])*bb;
        const int64_t last = int64_t(rowPtr_[br + 1])*bb;
        for (int64_t i = first; i < last; i++)
        {
            valuesD_[i] = 0.0;
        }
        for (int64_t q = blockRowFirst_[br]; q < blockRowFirst_[br + 1]; q++)
        {
            const int k = byBlockRow_[q];
            valuesD_[slot_[k]] += v[k];
        }
        if (mixed_)
        {
            for (int64_t i = first; i < last; i++)
            {
                valuesF_[i] = static_cast<float>(valuesD_[i]);
            }
        }
    }
}


void Foam::amgxCoupledSolver::rebuildSolver()
{
    if (solver_)
    {
        AMGX_solver_destroy(static_cast<AMGX_solver_handle>(solver_));
        solver_ = nullptr;
    }

    // The tolerance and the iteration cap go into a fresh config for the
    // solver, so a rebuild is where the adaptive inner tolerance takes
    // effect.  (AMGX_config_add_parameters refuses scoped keys on an
    // existing config, so the whole string is re-parsed instead.)
    if (solverCfg_)
    {
        AMGX_config_destroy(static_cast<AMGX_config_handle>(solverCfg_));
        solverCfg_ = nullptr;
    }
    std::ostringstream full;
    full.precision(6);
    full << config_.c_str()
         << ", main:tolerance=" << relTolerance_
         << ", main:max_iters=" << maxIter_;
    AMGX_config_handle cfg;
    check(AMGX_config_create(&cfg, full.str().c_str()), "config_create (solver)");
    solverCfg_ = static_cast<void*>(cfg);

    const AMGX_Mode mode = mixed_ ? AMGX_mode_dDFI : AMGX_mode_dDDI;
    AMGX_solver_handle slv;
    check
    (
        AMGX_solver_create
        (
            &slv,
            static_cast<AMGX_resources_handle>(rsrc_),
            mode,
            cfg
        ),
        "solver_create"
    );
    solver_ = static_cast<void*>(slv);
    check
    (
        AMGX_solver_setup(slv, static_cast<AMGX_matrix_handle>(mat_)),
        "solver_setup"
    );
}


Foam::label Foam::amgxCoupledSolver::solve
(
    const coupledLinearSystem& sys,
    scalarField& x
)
{
    AMGX_matrix_handle A = static_cast<AMGX_matrix_handle>(mat_);
    AMGX_vector_handle b = static_cast<AMGX_vector_handle>(rhs_);
    AMGX_vector_handle s = static_cast<AMGX_vector_handle>(sol_);

    using clock = std::chrono::steady_clock;
    const auto t0 = clock::now();

    if (!uploaded_)
    {
        buildPattern(sys);
    }
    else if (sys.nEntries() != nCoo_)
    {
        FatalErrorInFunction
            << "Sparsity pattern changed between outer iterations ("
            << sys.nEntries() << " triplets now, " << nCoo_ << " before)"
            << exit(FatalError);
    }

    fillValues(sys);
    const auto t1 = clock::now();

    const int n = static_cast<int>(nBlockRows_);
    const int nnzb = rowPtr_[nBlockRows_];
    const void* data =
        mixed_
      ? static_cast<const void*>(valuesF_.data())
      : static_cast<const void*>(valuesD_.data());

    if (!uploaded_)
    {
        check
        (
            AMGX_matrix_upload_all
            (
                A, n, nnzb, 4, 4, rowPtr_.data(), colIdx_.data(), data, nullptr
            ),
            "matrix_upload_all"
        );
        uploaded_ = true;
    }
    else
    {
        check
        (
            AMGX_matrix_replace_coefficients(A, n, nnzb, data, nullptr),
            "matrix_replace_coefficients"
        );
    }

    const auto t2 = clock::now();
    const bool rebuild = (nSolves_ % pcReuseInterval_ == 0);
    if (rebuild)
    {
        rebuildSolver();
    }
    else if (reuse_ == "resetup")
    {
        check
        (
            AMGX_solver_resetup(static_cast<AMGX_solver_handle>(solver_), A),
            "solver_resetup"
        );
    }
    nSolves_++;
    const auto t3 = clock::now();

    static_assert(sizeof(scalar) == sizeof(double), "scalar must be double");
    check(AMGX_vector_upload(b, n, 4, sys.rhs().cdata()), "vector_upload rhs");
    check(AMGX_vector_upload(s, n, 4, x.cdata()), "vector_upload x");

    AMGX_solver_handle slv = static_cast<AMGX_solver_handle>(solver_);
    check(AMGX_solver_solve(slv, b, s), "solver_solve");

    AMGX_SOLVE_STATUS status;
    AMGX_solver_get_status(slv, &status);
    int its = 0;
    AMGX_solver_get_iterations_number(slv, &its);
    if (status != AMGX_SOLVE_SUCCESS)
    {
        WarningInFunction
            << "AmgX solve did not converge (status " << int(status)
            << ") after " << its << " iterations" << endl;
    }

    check(AMGX_vector_download(s, x.data()), "vector_download");
    const auto t4 = clock::now();

    auto sec = [](clock::time_point a, clock::time_point b)
    {
        return std::chrono::duration<double>(b - a).count();
    };
    Info<< "AmgX timing: assemble " << sec(t0, t1) << " s, upload "
        << sec(t1, t2) << " s, " << (rebuild ? "setup " : "resetup ")
        << sec(t2, t3) << " s, solve " << sec(t3, t4) << " s" << endl;

    return static_cast<label>(its);
}

#endif // HAVE_AMGX

// ************************************************************************* //
