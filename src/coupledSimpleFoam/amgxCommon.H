/*---------------------------------------------------------------------------*\
  coupledSimpleFoam - Copyright (C) 2026 dhanninacs - GPL-3.0-or-later

Description
    One AmgX library session per process, shared by the coupled backend
    (amgxCoupledSolver) and the scalar lduMatrix solver (amgxLduSolver).
    Reference counted: every owner of AmgX handles acquires the library and
    releases it after destroying its handles; the last release calls
    AMGX_finalize.  Owners are OpenFOAM objects (the coupled backend, and a
    per-field state registered on the mesh), so all of them are destroyed
    before main returns - never from an exit handler, where the CUDA runtime
    and AmgX's own static memory pool may already be gone.

\*---------------------------------------------------------------------------*/

#ifndef amgxCommon_H
#define amgxCommon_H

#ifdef HAVE_AMGX

#include <amgx_c.h>
#include <string>
#include "word.H"
#include "label.H"

namespace Foam
{
namespace amgxCommon
{
    //- Take a reference on the library, initialising it on the first
    void acquire();

    //- Drop a reference; the last one destroys the shared resources and
    //  finalises the library
    void release();

    //- The one AmgX resources object of the process (one GPU), shared by
    //  every matrix, vector and solver: AmgX's device memory pool is per
    //  process, and destroying one of several resources objects while
    //  another's handles live aborts at exit ("Cuda failure: invalid
    //  argument").  Valid between acquire and the last release.
    AMGX_resources_handle resources();

    //- Let AmgX's own output through (off by default)
    void setVerbose(bool on);

    //- Fatal error when an AmgX call failed
    void check(AMGX_RC rc, const char* what);

    //- The AMG smoother part of the built-in configurations, by name:
    //  "DILU" (multicolour DILU, one pre/post sweep) or "GS" (multicolour
    //  Gauss-Seidel, relaxation 0.9, two pre/post sweeps: robust on
    //  stretched, high-aspect-ratio meshes, FoamWin #4). Anything else is
    //  a fatal error naming the choices.
    std::string smootherConfig(const word& smoother);

    //- Fatal error, with what to try, when a solution came back non-finite
    void checkFinite(const double* x, label n, const char* what);
}
}

#endif // HAVE_AMGX

#endif
