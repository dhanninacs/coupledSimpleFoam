/*---------------------------------------------------------------------------*\
  coupledSimpleFoam - Copyright (C) 2026 dhanninacs - GPL-3.0-or-later
\*---------------------------------------------------------------------------*/

#include "amgxLduSolver.H"

#ifdef HAVE_AMGX

#include <algorithm>
#include <sstream>
#include <string>
#include <vector>

#include "amgxCommon.H"
#include "addToRunTimeSelectionTable.H"
#include "regIOobject.H"
#include <map>
#include <cmath>
#include "objectRegistry.H"
#include "Time.H"

namespace Foam
{
    defineTypeNameAndDebug(amgxLduSolver, 0);

    lduMatrix::solver::addsymMatrixConstructorToTable<amgxLduSolver>
        addamgxLduSolverSymMatrixConstructorToTable_;

    lduMatrix::solver::addasymMatrixConstructorToTable<amgxLduSolver>
        addamgxLduSolverAsymMatrixConstructorToTable_;
}


namespace
{
    const char* const defaultScalarConfig =
        "config_version=2, exception_handling=1, "
        "solver(main)=PBICGSTAB, "
        "main:convergence=RELATIVE_INI_CORE, main:norm=L2, "
        "main:monitor_residual=1, main:print_solve_stats=0, "
        "main:preconditioner(amg)=AMG, "
        "amg:algorithm=AGGREGATION, amg:selector=SIZE_2, "
        "@SMOOTHER@, amg:max_iters=1, "
        "amg:cycle=V, amg:max_levels=50, "
        "amg:coarse_solver=DENSE_LU_SOLVER, amg:dense_lu_num_rows=64, "
        "amg:min_coarse_rows=32";
}


namespace Foam
{

//- One field's cached AmgX state: the CSR pattern of its ldu matrix, the
//  positions every ldu coefficient lands at, and the AmgX handles.  It is a
//  regIOobject held by the mesh's registry, so it is destroyed with the mesh
//  at the end of the run (before any exit handler) and releases its handles
//  and its reference on the AmgX library then.
class amgxFieldState
:
    public regIOobject
{
public:

    TypeName("amgxFieldState");

    int n = 0;
    int nFaces = 0;
    //- The addressing the pattern was built from: a topology change (a
    //  refiner, a mesh-to-mesh map) gives the matrix new addressing.
    const void* addressing = nullptr;
    int nnz = 0;
    std::vector<int> rowPtr;
    std::vector<int> colIdx;
    std::vector<int> diagPos;
    std::vector<int> upperPos;   // entry (l[f], u[f])
    std::vector<int> lowerPos;   // entry (u[f], l[f])
    std::vector<double> values;
    //- The values in single precision, for the mixed (dDFI) mode, whose
    //  matrix AmgX takes as float
    std::vector<float> valuesF;
    AMGX_config_handle rsrcCfg = nullptr;
    AMGX_resources_handle rsrc = nullptr;
    AMGX_matrix_handle A = nullptr;
    AMGX_vector_handle b = nullptr;
    AMGX_vector_handle x = nullptr;
    bool uploaded = false;
    //- dDDI (double) or dDFI (mixed: single-precision matrix and AMG)
    AMGX_Mode mode = AMGX_mode_dDDI;

    //- Solvers kept between solves, one per inner-tolerance decade (the
    //  tolerance lives in the AmgX config): built once, then refreshed with
    //  resetup on the new coefficients, and rebuilt in full every
    //  setupInterval solves.
    struct keptSolver
    {
        AMGX_config_handle cfg = nullptr;
        AMGX_solver_handle slv = nullptr;
        int sinceSetup = 0;
    };
    std::map<int, keptSolver> solvers;

    amgxFieldState
    (
        const IOobject& io,
        const lduMatrix& m,
        const std::string& config,
        const bool mixed
    );

    virtual ~amgxFieldState()
    {
        for (auto& kv : solvers)
        {
            if (kv.second.slv) AMGX_solver_destroy(kv.second.slv);
            if (kv.second.cfg) AMGX_config_destroy(kv.second.cfg);
        }
        if (x) AMGX_vector_destroy(x);
        if (b) AMGX_vector_destroy(b);
        if (A) AMGX_matrix_destroy(A);
        amgxCommon::release();
    }

    virtual bool writeData(Ostream&) const
    {
        return true;
    }
};

defineTypeNameAndDebug(amgxFieldState, 0);

} // End namespace Foam


Foam::amgxFieldState::amgxFieldState
(
    const IOobject& io,
    const lduMatrix& m,
    const std::string& config,
    const bool mixed
)
:
    regIOobject(io)
{
    mode = mixed ? AMGX_mode_dDFI : AMGX_mode_dDDI;
    amgxCommon::acquire();

    const lduAddressing& addr = m.lduAddr();
    n = static_cast<int>(addr.size());
    const labelUList& l = addr.lowerAddr();
    const labelUList& u = addr.upperAddr();
    nFaces = static_cast<int>(l.size());
    addressing = static_cast<const void*>(&addr);

    std::vector<int> count(n, 1);  // the diagonal
    for (int f = 0; f < nFaces; f++)
    {
        count[l[f]]++;
        count[u[f]]++;
    }
    rowPtr.assign(n + 1, 0);
    for (int i = 0; i < n; i++)
    {
        rowPtr[i + 1] = rowPtr[i] + count[i];
    }
    nnz = rowPtr[n];

    // Columns per row with an origin tag, sorted, so the position of every
    // ldu coefficient in the CSR value array can be recorded.
    std::vector<std::pair<int, int>> entries(nnz);  // (col, tag)
    std::vector<int> fill(rowPtr.begin(), rowPtr.end() - 1);
    for (int i = 0; i < n; i++)
    {
        entries[fill[i]++] = {i, -1};                 // diagonal
    }
    for (int f = 0; f < nFaces; f++)
    {
        entries[fill[l[f]]++] = {static_cast<int>(u[f]), 2*f};      // upper
        entries[fill[u[f]]++] = {static_cast<int>(l[f]), 2*f + 1};  // lower
    }
    colIdx.resize(nnz);
    diagPos.resize(n);
    upperPos.resize(nFaces);
    lowerPos.resize(nFaces);
    for (int i = 0; i < n; i++)
    {
        auto first = entries.begin() + rowPtr[i];
        auto last = entries.begin() + rowPtr[i + 1];
        std::sort(first, last);
        for (int k = rowPtr[i]; k < rowPtr[i + 1]; k++)
        {
            colIdx[k] = entries[k].first;
            const int tag = entries[k].second;
            if (tag < 0) diagPos[i] = k;
            else if (tag % 2 == 0) upperPos[tag/2] = k;
            else lowerPos[tag/2] = k;
        }
    }
    values.assign(nnz, 0.0);

    rsrc = amgxCommon::resources();
    amgxCommon::check(AMGX_matrix_create(&A, rsrc, mode), "matrix_create");
    amgxCommon::check(AMGX_vector_create(&b, rsrc, mode), "vector_create");
    amgxCommon::check(AMGX_vector_create(&x, rsrc, mode), "vector_create");
}


namespace
{
    //- The field's state, made and registered on the mesh on first use
    Foam::amgxFieldState& stateFor
    (
        const Foam::word& field,
        const Foam::lduMatrix& m,
        const std::string& config,
        const bool mixed
    )
    {
        const Foam::objectRegistry& db = m.mesh().db();
        const Foam::word name("amgxState:" + field);
        if (!db.foundObject<Foam::amgxFieldState>(name))
        {
            Foam::amgxFieldState* s = new Foam::amgxFieldState
            (
                Foam::IOobject
                (
                    name,
                    db.time().constant(),
                    db,
                    Foam::IOobject::NO_READ,
                    Foam::IOobject::NO_WRITE
                ),
                m,
                config,
                mixed
            );
            s->store();
        }
        Foam::amgxFieldState& s =
            db.lookupObjectRef<Foam::amgxFieldState>(name);
        const Foam::lduAddressing& addr = m.lduAddr();
        if
        (
            s.n != static_cast<int>(addr.size())
         || s.nFaces != static_cast<int>(addr.lowerAddr().size())
         || s.addressing != static_cast<const void*>(&addr)
         || s.mode != (mixed ? AMGX_mode_dDFI : AMGX_mode_dDDI)
        )
        {
            // The mesh changed topology (a refiner, mesh-to-mesh mapping):
            // drop the old pattern and GPU handles and build them anew for
            // the new addressing. The registry owns the state, so checking
            // it out deletes it.
            if (Foam::lduMatrix::debug)
            {
                Foam::Info<< "AmgX: rebuilding the matrix of " << field
                    << " for a changed mesh (" << s.n << " -> "
                    << addr.size() << " cells)" << Foam::endl;
            }
            s.checkOut();
            return stateFor(field, m, config, mixed);
        }
        return s;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::amgxLduSolver::amgxLduSolver
(
    const word& fieldName,
    const lduMatrix& matrix,
    const FieldField<Field, scalar>& interfaceBouCoeffs,
    const FieldField<Field, scalar>& interfaceIntCoeffs,
    const lduInterfaceFieldPtrsList& interfaces,
    const dictionary& solverControls
)
:
    lduMatrix::solver
    (
        fieldName,
        matrix,
        interfaceBouCoeffs,
        interfaceIntCoeffs,
        interfaces,
        solverControls
    )
{
    forAll(interfaces, i)
    {
        if (interfaces.set(i))
        {
            FatalErrorInFunction
                << "solver AmgX is serial and does not handle coupled"
                << " (processor or cyclic) interfaces; field " << fieldName
                << exit(FatalError);
        }
    }
    amgxCommon::setVerbose(solverControls.lookupOrDefault<bool>("amgxVerbose", false));
}


Foam::solverPerformance Foam::amgxLduSolver::solve
(
    scalarField& psi,
    const scalarField& source,
    const direction cmpt
) const
{
    solverPerformance solverPerf(typeName, fieldName_);

    const label nCells = psi.size();
    scalarField yA(nCells);
    scalarField pA(nCells);

    matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
    scalarField rA(source - yA);
    const scalar normFactor = this->normFactor(psi, source, yA, pA);

    solverPerf.initialResidual() =
        gSumMag(rA, matrix().mesh().comm())/normFactor;
    solverPerf.finalResidual() = solverPerf.initialResidual();

    if (minIter_ <= 0 && solverPerf.checkConvergence(tolerance_, relTol_))
    {
        return solverPerf;
    }

    const std::string config
    (
        controlDict_.lookupOrDefault<string>
        (
            "amgxConfig",
            string
            (
                std::string(defaultScalarConfig).replace
                (
                    std::string(defaultScalarConfig).find("@SMOOTHER@"), 10,
                    amgxCommon::smootherConfig(controlDict_.lookupOrDefault<word>("amgxSmoother", "GS"))
                )
            )
        )
    );
    const bool mixed =
        controlDict_.lookupOrDefault<word>("amgxPrecision", "double") == "mixed";
    if (mixed)
    {
        FatalErrorInFunction
            << "amgxPrecision mixed is not available: with AmgX 2.5 the mixed (dDFI) "
                << "mode fails inside AmgX (a floating-point exception in its convergence "
                << "check, or \"Configuration feature is not implemented\") on the cases "
                << "tested. Use amgxPrecision double (the default)."
            << exit(FatalError);
    }
    amgxFieldState& s = stateFor(fieldName_, matrix_, config, mixed);

    // Coefficients into the cached CSR positions.
    const scalarField& diag = matrix_.diag();
    for (int i = 0; i < s.n; i++)
    {
        s.values[s.diagPos[i]] = diag[i];
    }
    if (matrix_.hasUpper())
    {
        const scalarField& upper = matrix_.upper();
        const scalarField& lower = matrix_.lower();   // == upper if symmetric
        forAll(upper, f)
        {
            s.values[s.upperPos[f]] = upper[f];
            s.values[s.lowerPos[f]] = lower[f];
        }
    }

    const void* vals = s.values.data();
    if (s.mode == AMGX_mode_dDFI)
    {
        s.valuesF.resize(s.values.size());
        for (size_t i = 0; i < s.values.size(); i++)
        {
            s.valuesF[i] = static_cast<float>(s.values[i]);
        }
        vals = s.valuesF.data();
    }

    if (!s.uploaded)
    {
        amgxCommon::check
        (
            AMGX_matrix_upload_all
            (
                s.A, s.n, s.nnz, 1, 1,
                s.rowPtr.data(), s.colIdx.data(), vals, nullptr
            ),
            "matrix_upload_all"
        );
        s.uploaded = true;
    }
    else
    {
        amgxCommon::check
        (
            AMGX_matrix_replace_coefficients
            (
                s.A, s.n, s.nnz, vals, nullptr
            ),
            "matrix_replace_coefficients"
        );
    }

    // OpenFOAM stops at  final <= max(tolerance, relTol*initial)  in its
    // normalised L1 measure; AmgX is told the matching fraction of its own
    // (L2, initial-relative) residual, and the host re-checks below.
    const scalar target =
        max(tolerance_, relTol_*solverPerf.initialResidual());
    scalar amgxTol =
        min(scalar(0.1), 0.5*target/max(solverPerf.initialResidual(), vSmall));

    amgxCommon::check
    (
        AMGX_vector_upload(s.b, s.n, 1, source.cdata()), "vector_upload rhs"
    );

    const label setupInterval =
        max(label(1), controlDict_.lookupOrDefault<label>("amgxSetupInterval", 10));

    label totalIts = 0;
    for (int pass = 0; pass < 4; pass++)
    {
        // The tolerance rounded down to a decade (never looser than asked),
        // so that the solver built for it serves the next solves too.
        const int decade =
            static_cast<int>(std::floor(std::log10(max(amgxTol, scalar(1e-14)))));
        amgxFieldState::keptSolver& k = s.solvers[decade];
        if (!k.slv)
        {
            std::ostringstream full;
            full.precision(6);
            full << config << ", main:tolerance=" << std::pow(10.0, decade)
                 << ", main:max_iters=" << max(maxIter_, label(1));
            amgxCommon::check(AMGX_config_create(&k.cfg, full.str().c_str()), "config_create");
            amgxCommon::check
            (
                AMGX_solver_create(&k.slv, s.rsrc, s.mode, k.cfg), "solver_create"
            );
            amgxCommon::check(AMGX_solver_setup(k.slv, s.A), "solver_setup");
            k.sinceSetup = 0;
        }
        else if (++k.sinceSetup >= setupInterval)
        {
            amgxCommon::check(AMGX_solver_setup(k.slv, s.A), "solver_setup");
            k.sinceSetup = 0;
        }
        else
        {
            // Same pattern, new coefficients: keep the hierarchy's structure.
            amgxCommon::check(AMGX_solver_resetup(k.slv, s.A), "solver_resetup");
        }
        amgxCommon::check
        (
            AMGX_vector_upload(s.x, s.n, 1, psi.cdata()), "vector_upload x"
        );
        amgxCommon::check(AMGX_solver_solve(k.slv, s.b, s.x), "solver_solve");
        int its = 0;
        AMGX_solver_get_iterations_number(k.slv, &its);
        totalIts += its;
        amgxCommon::check(AMGX_vector_download(s.x, psi.data()), "vector_download");
        amgxCommon::checkFinite(psi.cdata(), psi.size(), fieldName_.c_str());

        matrix_.Amul(yA, psi, interfaceBouCoeffs_, interfaces_, cmpt);
        rA = source - yA;
        solverPerf.finalResidual() =
            gSumMag(rA, matrix().mesh().comm())/normFactor;
        if
        (
            solverPerf.checkConvergence(tolerance_, relTol_)
         || totalIts >= maxIter_
        )
        {
            break;
        }
        // Short of OpenFOAM's criterion: continue from here, tighter.
        amgxTol = max(amgxTol*0.1, scalar(1e-14));
    }

    solverPerf.nIterations() = totalIts;
    return solverPerf;
}

#endif // HAVE_AMGX

// ************************************************************************* //
