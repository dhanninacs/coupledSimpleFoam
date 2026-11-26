/*---------------------------------------------------------------------------*\
  coupledSimpleFoam - Copyright (C) 2026 dhanninacs - GPL-3.0-or-later
\*---------------------------------------------------------------------------*/

#include "amgxCommon.H"

#ifdef HAVE_AMGX

#include <string>

#include "IOstreams.H"
#include "error.H"

namespace
{
    int refs_ = 0;
    bool verbose_ = false;
    AMGX_config_handle rsrcCfg_ = nullptr;
    AMGX_resources_handle rsrc_ = nullptr;

    void printCallback(const char* msg, int length)
    {
        if (verbose_)
        {
            Foam::Info<< std::string(msg, length).c_str();
        }
    }
}


void Foam::amgxCommon::acquire()
{
    if (refs_++ == 0)
    {
        check(AMGX_initialize(), "initialize");
        AMGX_register_print_callback(&printCallback);
        check
        (
            AMGX_config_create(&rsrcCfg_, "config_version=2"),
            "config_create (resources)"
        );
        check(AMGX_resources_create_simple(&rsrc_, rsrcCfg_), "resources_create");
    }
}


void Foam::amgxCommon::release()
{
    if (refs_ > 0 && --refs_ == 0)
    {
        AMGX_resources_destroy(rsrc_);
        AMGX_config_destroy(rsrcCfg_);
        rsrc_ = nullptr;
        rsrcCfg_ = nullptr;
        AMGX_finalize();
    }
}


AMGX_resources_handle Foam::amgxCommon::resources()
{
    return rsrc_;
}


void Foam::amgxCommon::setVerbose(bool on)
{
    verbose_ = verbose_ || on;
}


void Foam::amgxCommon::check(AMGX_RC rc, const char* what)
{
    if (rc != AMGX_RC_OK)
    {
        char buf[512];
        AMGX_get_error_string(rc, buf, sizeof(buf));
        FatalErrorInFunction
            << "AmgX " << what << " failed: " << buf << exit(FatalError);
    }
}

#endif // HAVE_AMGX
