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
    A verbatim copy is distributed in the LICENSE file at the root of this
    repository.

Description
    PETSc (MPI) backend for the coupled p-U system.  The whole translation
    unit compiles to nothing when the build did not find PETSc.

\*---------------------------------------------------------------------------*/

#include "petscCoupledSolver.H"

#ifdef HAVE_PETSC

#include <vector>
#include <petscksp.h>

#include "IOstreams.H"
#include "Pstream.H"
#include "error.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace
{
    bool petscInitialisedHere_ = false;
}

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::petscCoupledSolver::petscCoupledSolver(const dictionary& dict)
:
    name_("PETSc"),
    // Tight by default: the coupled outer loop is a Picard iteration, so the
    // linear system must be solved essentially exactly for the iteration
    // count to match the serial direct (Eigen SparseLU) reference.
    tolerance_(dict.lookupOrDefault<scalar>("tolerance", 1e-30)),
    relTolerance_(dict.lookupOrDefault<scalar>("relTol", 1e-12)),
    maxIter_(dict.lookupOrDefault<label>("maxLinearIter", 5000)),
    petscOptions_
    (
        dict.lookupOrDefault<string>
        (
            "petscOptions",
            // Fieldsplit-Schur with hypre BoomerAMG on the Schur block: the
            // only configuration that is both fast and robust on every mesh
            // topology tested.  On the 2-D periodic interrupted-fin channel
            // (stripChannel) it converges in 25 outer iterations and matches
            // the serial direct reference pressure drop to all reported
            // digits, where Schur+GAMG diverges; on the 483k-cell
            // snappyHexMesh interrupted-fin unit cell it takes 24 outer
            // iterations at ~3.8 s per outer iteration against 12.2 s with
            // GAMG on the Schur block.  Alternatives:
            //   ASM/ILU        robust in 2-D, does not scale to the 3-D
            //                  unstructured unit cell
            //   Schur+GAMG     fast on cavity and 3-D meshes, diverges on
            //                  the interrupted-fin channel
            //   Schur+hypre    fast and robust on all cases tested
            //                  (default)
            string
            (
                "-ksp_type fgmres "
                "-pc_type fieldsplit -pc_fieldsplit_type schur "
                "-pc_fieldsplit_schur_fact_type full "
                "-pc_fieldsplit_schur_precondition selfp "
                "-pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 "
                "-fieldsplit_0_ksp_type preonly "
                "-fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu "
                "-fieldsplit_1_ksp_type preonly "
                "-fieldsplit_1_pc_type hypre "
                "-fieldsplit_1_pc_hypre_type boomeramg"
            )
        )
    ),
    matPtr_(nullptr),
    rhsPtr_(nullptr),
    solPtr_(nullptr),
    kspPtr_(nullptr),
    built_(false),
    nCoo_(-1),
    reusePC_(dict.lookupOrDefault<bool>("reusePreconditioner", false))
{
    PetscBool inited = PETSC_FALSE;
    PetscInitialized(&inited);

    if (!inited)
    {
        // GPU: -use_gpu_aware_mpi is read by PetscInitialize itself, so it
        // cannot be supplied through petscOptions (which is inserted after).
        // Most system MPI installations are not CUDA-aware, including the
        // OpenMPI that OpenFOAM is commonly built against, so default it
        // off; a CPU-only PETSc simply ignores the option.  Set it back to
        // 1 through PETSC_OPTIONS on a CUDA-aware MPI stack.
        // PetscOptionsSetValue is explicitly allowed before
        // PetscInitialize.
        PetscOptionsSetValue(nullptr, "-use_gpu_aware_mpi", "0");

        // PETSc refuses PETSC_COMM_WORLD assignment before MPI_Init, so only
        // hand it OpenFOAM's communicator when OpenFOAM has already started
        // MPI (parallel run).  In serial PETSc initialises MPI itself.
        int mpiInited = 0;
        MPI_Initialized(&mpiInited);

        if (mpiInited)
        {
            PETSC_COMM_WORLD = MPI_COMM_WORLD;
        }

        const PetscErrorCode ierr = PetscInitializeNoArguments();

        if (ierr)
        {
            FatalErrorInFunction
                << "PetscInitialize failed with error code " << label(ierr)
                << exit(FatalError);
        }

        petscInitialisedHere_ = true;
    }

    if (petscOptions_.size())
    {
        PetscOptionsInsertString(nullptr, petscOptions_.c_str());
    }
}


Foam::petscCoupledSolver::~petscCoupledSolver()
{
    if (built_)
    {
        Mat A = static_cast<Mat>(matPtr_);
        Vec b = static_cast<Vec>(rhsPtr_);
        Vec s = static_cast<Vec>(solPtr_);
        KSP ksp = static_cast<KSP>(kspPtr_);

        KSPDestroy(&ksp);
        VecDestroy(&s);
        VecDestroy(&b);
        MatDestroy(&A);

        built_ = false;
    }

    if (petscInitialisedHere_)
    {
        PetscFinalize();
        petscInitialisedHere_ = false;
    }
}


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

Foam::label Foam::petscCoupledSolver::solve
(
    const coupledLinearSystem& sys,
    scalarField& x
)
{
    const PetscInt nLocal = static_cast<PetscInt>(sys.nLocalRows());
    const PetscInt nGlobal = static_cast<PetscInt>(sys.nGlobalRows());

    Mat A;
    Vec b;
    Vec s;
    KSP ksp;

    if (!built_)
    {
        MatCreate(PETSC_COMM_WORLD, &A);
        MatSetSizes(A, nLocal, nLocal, nGlobal, nGlobal);
        MatSetType(A, Pstream::parRun() ? MATMPIAIJ : MATSEQAIJ);

        // The unknowns are interleaved [u,v,w,p] per cell, i.e. a genuine
        // block size of 4.  Telling PETSc lets PCFIELDSPLIT build the
        // velocity/pressure index sets from options alone:
        //   -pc_type fieldsplit -pc_fieldsplit_type schur
        //   -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3
        MatSetBlockSize(A, static_cast<PetscInt>(sys.blockSize()));

        // MatSetFromOptions comes AFTER MatSetType, so the type set above is
        // only a default: -mat_type aijcusparse (GPU) or -mat_type baij etc.
        // in petscOptions replaces it.  Same for -vec_type cuda below.
        MatSetFromOptions(A);

        // COO assembly path: register the (static) sparsity pattern once.
        // PETSc copies the index arrays and sums duplicate (i,j) pairs
        // natively; per-iteration assembly is then a single
        // MatSetValuesCOO call over the value array.
        {
            nCoo_ = sys.nEntries();
            std::vector<PetscInt> ci(nCoo_);
            std::vector<PetscInt> cj(nCoo_);
            for (label k = 0; k < nCoo_; k++)
            {
                ci[k] = static_cast<PetscInt>(sys.row(k));
                cj[k] = static_cast<PetscInt>(sys.col(k));
            }
            MatSetPreallocationCOO
            (
                A,
                static_cast<PetscCount>(nCoo_),
                ci.data(),
                cj.data()
            );
        }

        VecCreate(PETSC_COMM_WORLD, &b);
        VecSetSizes(b, nLocal, nGlobal);
        VecSetBlockSize(b, static_cast<PetscInt>(sys.blockSize()));
        VecSetFromOptions(b);
        VecDuplicate(b, &s);

        KSPCreate(PETSC_COMM_WORLD, &ksp);
        KSPSetFromOptions(ksp);

        if (reusePC_)
        {
            // Build the preconditioner on the first solve, then freeze it
            // for all subsequent outer iterations (Picard updates change
            // the operator slowly, so a stale preconditioner stays useful).
            KSPSetReusePreconditioner(ksp, PETSC_TRUE);
        }

        matPtr_ = static_cast<void*>(A);
        rhsPtr_ = static_cast<void*>(b);
        solPtr_ = static_cast<void*>(s);
        kspPtr_ = static_cast<void*>(ksp);

        built_ = true;
    }
    else
    {
        A = static_cast<Mat>(matPtr_);
        b = static_cast<Vec>(rhsPtr_);
        s = static_cast<Vec>(solPtr_);
        ksp = static_cast<KSP>(kspPtr_);
    }

    if (sys.nEntries() != nCoo_)
    {
        FatalErrorInFunction
            << "Sparsity pattern changed between outer iterations ("
            << sys.nEntries() << " triplets now, " << nCoo_
            << " registered with MatSetPreallocationCOO)"
            << exit(FatalError);
    }

    // Foam::scalar and PetscScalar must both be double for the direct
    // value-array handoff (real, double-precision PETSc build)
    static_assert
    (
        sizeof(scalar) == sizeof(PetscScalar),
        "scalar/PetscScalar size mismatch - COO handoff invalid"
    );

    MatSetValuesCOO
    (
        A,
        reinterpret_cast<const PetscScalar*>(sys.valData()),
        INSERT_VALUES
    );

    // RHS and initial guess: local rows are contiguous in the PETSc
    // layout, so write the arrays directly.
    //
    // GPU: VecGetArrayWrite, not VecGetArray.  Every local entry is
    // overwritten below, so the device->host fetch that VecGetArray performs
    // is pure waste on a VECCUDA - two full vector copies per outer
    // iteration.  Semantics are identical on a host Vec.
    {
        PetscScalar* bArr;
        PetscScalar* sArr;
        VecGetArrayWrite(b, &bArr);
        VecGetArrayWrite(s, &sArr);
        const scalarField& rhsF = sys.rhs();
        for (PetscInt i = 0; i < nLocal; i++)
        {
            bArr[i] = rhsF[i];
            sArr[i] = x[i];
        }
        VecRestoreArrayWrite(b, &bArr);
        VecRestoreArrayWrite(s, &sArr);
    }

    KSPSetOperators(ksp, A, A);
    KSPSetTolerances
    (
        ksp,
        relTolerance_,
        tolerance_,
        PETSC_DEFAULT,
        static_cast<PetscInt>(maxIter_)
    );
    KSPSetInitialGuessNonzero(ksp, PETSC_TRUE);
    KSPSetFromOptions(ksp);

    KSPSolve(ksp, b, s);

    KSPConvergedReason reason;
    KSPGetConvergedReason(ksp, &reason);

    PetscInt its = 0;
    KSPGetIterationNumber(ksp, &its);

    if (reason < 0)
    {
        WarningInFunction
            << "PETSc KSP diverged, reason " << label(reason)
            << " after " << label(its) << " iterations" << endl;
    }

    const PetscScalar* sArray;
    VecGetArrayRead(s, &sArray);
    for (PetscInt i = 0; i < nLocal; i++)
    {
        x[i] = sArray[i];
    }
    VecRestoreArrayRead(s, &sArray);

    return static_cast<label>(its);
}

#endif // HAVE_PETSC

// ************************************************************************* //
