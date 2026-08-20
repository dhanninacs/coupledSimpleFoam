# Installing coupledSimpleFoam

This guide starts from a machine with none of the software installed and ends
with a verified `coupledSimpleFoam` binary that reproduces the shipped
lid-driven cavity tutorial. It covers:

- **(a)** native Ubuntu 22.04 / 24.04
- **(b)** Windows with WSL2 running an Ubuntu distribution
- **(c)** other Linux distributions, in outline

The solver targets the **OpenFOAM Foundation (openfoam.org) release 14** API.

## How to read the steps

Every step is marked with when it has to be run:

| Marker | Meaning |
| --- | --- |
| **[one-off]** | Run once per machine. Installing packages, building the solver. |
| **[per-shell]** | Run in every new terminal, or put it in `~/.bashrc` so the shell does it. Sourcing the OpenFOAM environment is the only per-shell step the solver needs. |

The only genuinely per-shell requirement is
`. /opt/openfoam14/etc/bashrc`. Everything else - including the build
switches in section 6 - is a one-off, or a per-command environment variable.

---

## 1. Prerequisites at a glance

| Component | Minimum version | Required | Purpose |
| --- | --- | --- | --- |
| OpenFOAM (Foundation, openfoam.org) | 14 | Required | Mesh, field, `fvMatrix` and `momentumTransport` libraries; the `wmake` build system. The solver is written against the version 14 API. |
| C++ compiler | Whatever OpenFOAM 14 requires (GCC 9 or newer) | Required | Pulled in by the OpenFOAM package. |
| GNU make | 4.x | Required | `wmake` drives it, and `Make/options` is read raw by GNU make. |
| Eigen | 3.4 (3.4.0 tested) | Required | Serial linear-algebra backend (`SparseLU`, `BiCGSTAB` + `IncompleteLUT`). Header-only; nothing to link. |
| pkg-config | any | Required only for PETSc autodetection | `Make/options` locates PETSc through it. |
| PETSc | 3.19 (3.19.6 and 3.25.4 tested) | Optional | The MPI backend (`MATMPIAIJ` + `KSP`). Required for any parallel run; strongly recommended for 3-D meshes, where a serial direct factorisation is uneconomical. |
| hypre | As configured into PETSc | Optional, recommended with PETSc | BoomerAMG on the Schur block - the shipped default preconditioner. Ubuntu's `petsc-dev` includes it. |
| MPI | The same implementation OpenFOAM uses (OpenMPI 4.1 tested) | Optional (needed with PETSc) | Parallel runs. PETSc and OpenFOAM **must** agree; see section 4.2. |
| Python 3 | 3.8 | Optional | `tools/compare_fields.py`, the field-comparison utility used by the tutorials. |

Rough disk and time budget:

| Item | Disk | Time |
| --- | --- | --- |
| OpenFOAM 14 from apt | ~2 GB | 5-10 min |
| Eigen + PETSc from apt | ~500 MB | 1-2 min |
| The solver itself | ~50 MB | 1-2 min |

---

## 2. Install OpenFOAM 14

### 2.1 Add the Foundation apt repository **[one-off]**

These commands are identical on native Ubuntu and inside a WSL2 Ubuntu
distribution.

```bash
sudo apt-get update
sudo apt-get install -y wget software-properties-common

# Repository signing key, as an ASCII-armoured key in the trusted keyring dir
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"

# Repository.  add-apt-repository fills in the release codename automatically.
sudo add-apt-repository "http://dl.openfoam.org/ubuntu main dev"

sudo apt-get update
sudo apt-get install -y openfoam14
```

The package installs into `/opt/openfoam14`.

### 2.2 Supported releases and the amd64 caveat

The Foundation publishes packages for the current Ubuntu LTS releases. Before
running the install, confirm that a candidate exists for this machine:

```bash
lsb_release -cs                 # the codename apt will look for
apt-cache policy openfoam14     # "Candidate: (none)" means no package for it
```

Two things go wrong on newer releases:

- **amd64 only.** The Foundation builds for `amd64`. On `arm64` - which
  includes WSL2 on an ARM Windows device, and ARM Linux generally - there is
  no candidate, and OpenFOAM 14 has to be
  [built from source](https://openfoam.org/download/source/) instead.
- **Multi-architecture systems.** `add-apt-repository` on Ubuntu 24.04 and
  newer writes an entry that apt applies to *every* enabled architecture, so if
  `i386` or `arm64` has been enabled with `dpkg --add-architecture`,
  `apt-get update` reports a 404 for the architecture the repository does not
  carry. Pin the entry instead:

  ```bash
  sudo add-apt-repository -r "http://dl.openfoam.org/ubuntu main dev"
  sudo sh -c 'echo "deb [arch=amd64] http://dl.openfoam.org/ubuntu $(lsb_release -cs) main dev" > /etc/apt/sources.list.d/openfoam.list'
  sudo apt-get update
  ```

If no package exists for the release in use, the two alternatives are building
OpenFOAM 14 from source, or running a container image published by
openfoam.org. Both provide an `etc/bashrc` to source, exactly as the package
does - only its path differs (a source build puts it under the build tree, for
example `$HOME/OpenFOAM/OpenFOAM-14/etc/bashrc`; a container has it at
`/opt/openfoam14/etc/bashrc` inside the container). Substitute that path
wherever this guide writes `/opt/openfoam14/etc/bashrc`; nothing else changes.

### 2.3 Source the environment **[per-shell]**

```bash
. /opt/openfoam14/etc/bashrc
```

To have every new interactive shell do it **[one-off]**:

```bash
echo '. /opt/openfoam14/etc/bashrc' >> ~/.bashrc
```

Sourcing sets `WM_PROJECT_DIR`, `WM_PROJECT_VERSION`, `FOAM_USER_APPBIN`,
`FOAM_TUTORIALS` and puts the OpenFOAM applications on `PATH`. Every script in
this repository - `Allwmake`, `Allrun`, `Allclean` - refuses to run without it
and says so.

Non-interactive shells (`ssh host 'command'`, `sh -c`, CI jobs, cron) do not
read `~/.bashrc`, so scripts should source the file explicitly rather than rely
on the login shell.

### 2.4 Verify **[one-off]**

```bash
echo "$WM_PROJECT_VERSION"        # -> 14
echo "$FOAM_USER_APPBIN"          # -> <home>/OpenFOAM/<user>-14/platforms/<arch>/bin
which blockMesh foamRun           # -> /opt/openfoam14/platforms/.../bin/...
foamRun -help | head -3
```

A full smoke test of the OpenFOAM installation itself, independent of this
solver:

```bash
mkdir -p "$HOME/openfoam-check" && cd "$HOME/openfoam-check"
cp -r "$FOAM_TUTORIALS/incompressibleFluid/pitzDaily" .
cd pitzDaily && blockMesh && foamRun
```

Note that Foundation OpenFOAM 14 has **no standalone `simpleFoam`**: the
segregated reference solver is `foamRun -solver incompressibleFluid`, and
tutorial `blockMeshDict` files live under
`$FOAM_TUTORIALS/resources/blockMesh/`.

### 2.5 Foundation, not ESI

**This solver targets the Foundation (openfoam.org) OpenFOAM 14 API and does
not compile against ESI (openfoam.com, `vNNNN`) releases without
modification.** The two lines diverged years ago, and the divergence is in
exactly the places this solver touches:

- the momentum-transport framework is `momentumTransportModel` with
  `constant/momentumTransport` and the method `divDevSigma`, against ESI's
  `turbulenceModel` with `constant/turbulenceProperties` and `divDevReff`
- viscosity comes from `constant/physicalProperties` through
  `physicalProperties` / `viscosityModel`, against ESI's
  `constant/transportProperties`
- `fvCFD.H` does not exist in Foundation 14, so the sources include
  `argList.H`, `Time.H`, `fvMesh.H`, `volFields.H`, `surfaceFields.H`, `fvm.H`,
  `fvc.H` and `fvMatrices.H` individually
- several small API differences (`dimViscosity`, `fvPatch::patch()`,
  `Time::constantPath()`) that exist in one line and not the other

Porting is a contained job - the assembly itself only uses `fvMatrix`
coefficient access, `syncTools` and `globalIndex`, all of which exist in both
lines - but it is not a no-op, and no ESI build is tested here. `Allwmake`
prints a warning if `WM_PROJECT_VERSION` is not 14.

---

## 3. Install Eigen **[one-off]**

Eigen is header-only: it is a compile-time dependency only, there is nothing to
link and nothing to ship at run time. It provides the serial backends
(`SparseLU`, and `BiCGSTAB` with an `IncompleteLUT` preconditioner) and is
always required, including in a PETSc build.

```bash
sudo apt-get install -y libeigen3-dev
```

The headers land in `/usr/include/eigen3`, which is the default the build
expects. Verify:

```bash
ls /usr/include/eigen3/Eigen/Sparse      # must exist
```

If Eigen lives somewhere else - a manual install, a module system, a non-Debian
distribution - pass its location through `EIGEN_INC` at build time:

```bash
EIGEN_INC=-I/opt/eigen/include ./Allwmake
```

---

## 4. Install PETSc (optional, for MPI)

PETSc is what makes parallel runs possible. Without it the solver builds and
runs perfectly well in serial with the Eigen backends, and exits with a clear
error if a parallel run is attempted. With 3-D meshes PETSc is effectively
mandatory, because the memory and time cost of a serial sparse direct
factorisation grows very quickly.

### 4.1 The distribution package route **[one-off]**

```bash
sudo apt-get install -y petsc-dev
```

On Ubuntu 22.04 / 24.04 this installs a real-scalar PETSc (3.19.6 on 24.04)
built against the system OpenMPI, with hypre included - which matters, because
the shipped default preconditioner puts hypre BoomerAMG on the Schur block.

Confirm that `pkg-config` can see it - this is exactly how `Make/options`
detects PETSc:

```bash
pkg-config --exists PETSc && echo "found as PETSc"
pkg-config --exists petsc && echo "found as petsc"
pkg-config --modversion PETSc
pkg-config --libs PETSc
```

The build tries the name `PETSc` first (Debian/Ubuntu spelling) and then
`petsc` (the source-build spelling), so either is fine. If neither is found,
see the troubleshooting entry on `PKG_CONFIG_PATH`.

Check that hypre is present in the PETSc that was found:

```bash
pkg-config --libs PETSc | tr ' ' '\n' | grep -i hypre
```

If that comes back empty, the solver still runs, but the default
`petscOptions` string asks for `-fieldsplit_1_pc_type hypre` and the run fails
in the linear solve. Either rebuild PETSc with `--download-hypre`, or override
`petscOptions` in the case (`doc/USAGE.md` documents the alternatives; the
GAMG variant needs no hypre).

### 4.2 MPI consistency: PETSc and OpenFOAM must use the same MPI

This is the single most common way a PETSc-enabled build goes wrong. OpenFOAM,
PETSc and the solver binary all link an MPI library. If two different MPI
implementations - or two different builds of the same one - end up in one
process, the failure is not a clean error message: typical symptoms are a crash
inside `MPI_Init`, an "MPI_Init invoked multiple times" abort, unresolved
`ompi_mpi_comm_world`-style symbols at link time, or (the confusing one) an
`mpirun -np 4` in which every rank reports itself as rank 0 of 1 and the run
silently computes four independent serial cases.

Check which MPI OpenFOAM uses, and which one PETSc links **[one-off]**:

```bash
echo "$WM_MPLIB"                    # SYSTEMOPENMPI on an apt OpenFOAM
which mpicc mpirun
mpicc --showme:link                 # OpenMPI: the -L/-l set the wrapper adds
mpicc --showme:compile              # OpenMPI: the -I set the wrapper adds

pkg-config --libs PETSc | tr ' ' '\n' | grep -iE 'mpi|openmpi|mpich'
```

Both should point at the same installation - on Ubuntu, the system OpenMPI
under `/usr/lib/x86_64-linux-gnu/openmpi`. After building the solver, the
authoritative check is what the binary actually loads:

```bash
ldd "$FOAM_USER_APPBIN/coupledSimpleFoam" | grep -iE 'mpi|petsc'
```

Exactly one `libmpi.so*` should appear. Two, or one `libmpi.so` alongside one
`libmpich.so`, is the mismatch.

Two related notes:

- PETSc's `.pc` file does not carry the MPI include path, so `Make/options`
  takes it from `mpicc --showme:compile` and `mpicc --showme:link`. `--showme`
  is an OpenMPI extension. On an MPICH stack those come back empty and the MPI
  include and library flags have to be supplied by hand through `EXE_INC` /
  `EXE_LIBS`, or by putting an `mpicc` wrapper that understands `--showme` on
  `PATH`.
- The same requirement applies to a PETSc built from source: configure it with
  `--with-cc=/usr/bin/mpicc` and `--with-cxx=/usr/bin/mpicxx`, the same MPI
  OpenFOAM uses, rather than letting PETSc pull in its own with
  `--download-mpich` or `--download-openmpi`.

---

## 5. WSL2 specifics

Everything above applies unchanged inside a WSL2 Ubuntu distribution. Two
things are different, and getting either of them wrong is expensive.

### 5.1 Build in the Linux filesystem, not under `/mnt/c`

Compiling a tree that lives on a mounted Windows drive is roughly an order of
magnitude slower than compiling in the Linux filesystem, because every one of
the thousands of small header reads crosses the 9P filesystem bridge. Keep the
checkout under the Linux home directory:

```bash
git clone https://github.com/dhanninacs/coupledSimpleFoam.git "$HOME/coupledSimpleFoam"
cd "$HOME/coupledSimpleFoam"
```

If the repository has to live on the Windows side for editing convenience, copy
it into `$HOME` for the build rather than building in place. The same applies
to running cases: mesh generation and I/O on `/mnt/c` are slow.

### 5.2 `mpirun` drains standard input

This is not specific to WSL2, but it is where it usually bites, because runs
there are often driven by a script piped into a shell. `mpirun` consumes
standard input, so such a script loses everything after the first `mpirun`
unless each one is given `< /dev/null`. Every `mpirun` in this guide carries
it for that reason.

---

## 6. Build the solver

### 6.1 Default build **[one-off]**

```bash
. /opt/openfoam14/etc/bashrc          # [per-shell] if not in ~/.bashrc
cd /path/to/coupledSimpleFoam
./Allwmake
```

`Allwmake` checks the environment, prints the configuration it resolved, and
runs `wmake src/coupledSimpleFoam`. It passes any arguments straight through to
`wmake`, so `./Allwmake -j 8` works. The configuration echo looks like:

```
coupledSimpleFoam build configuration
    OpenFOAM         : 14 (/opt/openfoam14)
    Eigen include    : -I/usr/include/eigen3 (default)
    PETSc backend    : enabled (found by pkg-config)
```

**Read that PETSc line.** It is the difference between a build that can run in
parallel and one that cannot, and it is decided at compile time.

### 6.2 Build switches

All switches are environment variables; nothing is baked into the repository.

| Switch | Effect |
| --- | --- |
| *(none)* | PETSc is auto-detected through `pkg-config` (`PETSc`, then `petsc`) and enabled if found. This is the normal build. |
| `COUPLED_NO_PETSC=1` | Force a serial, Eigen-only build even when PETSc is installed. Useful for a machine that will only ever run serial cases, and for isolating a PETSc problem. |
| `EIGEN_INC` | Include flag for the Eigen headers, for example `EIGEN_INC=-I/opt/eigen/include`. Defaults to `-I/usr/include/eigen3`. |

Examples:

```bash
./Allwmake                            # default: PETSc if pkg-config finds it
COUPLED_NO_PETSC=1 ./Allwmake         # serial Eigen only
```

**Both produce the same executable name and overwrite each other.** Build the
variant that is going to be run. When switching between variants, clean first -
the include paths and the `-DHAVE_PETSC` define differ, and a stale object tree
produces a binary that does not match the switch:

```bash
./Allclean && COUPLED_NO_PETSC=1 ./Allwmake
```

(`./Allclean` also removes tutorial run output. To clean only the build,
`wclean src/coupledSimpleFoam`.)

### 6.3 Where the binary lands, and confirming it is on `PATH`

`wmake` installs into OpenFOAM's per-user application directory:

```
$FOAM_USER_APPBIN/coupledSimpleFoam
```

which is `~/OpenFOAM/<user>-14/platforms/<arch>/bin/coupledSimpleFoam`.
Sourcing `etc/bashrc` puts that directory on `PATH`, so no install step and no
`sudo` is involved.

```bash
ls -l "$FOAM_USER_APPBIN/coupledSimpleFoam"
command -v coupledSimpleFoam          # must resolve, and to that same path
coupledSimpleFoam -help | head -5
```

Check which PETSc the binary actually picked up:

```bash
ldd "$FOAM_USER_APPBIN/coupledSimpleFoam" | grep -iE 'petsc|mpi'
```

- Default build: one `libpetsc*.so` from the distribution library path.
- `COUPLED_NO_PETSC=1` build: no `libpetsc` at all.

If `command -v coupledSimpleFoam` finds nothing while the file exists, the
OpenFOAM environment is not sourced in that shell - see section 2.3.

---

## 7. Verify the installation

The lid-driven cavity tutorial is the correct acceptance test: it is small,
it runs in serial with no PETSc, and its converged outer-iteration count is a
fixed, reproducible number.

```bash
. /opt/openfoam14/etc/bashrc          # [per-shell]
cd /path/to/coupledSimpleFoam
cd tutorials/cavity
./Allrun
```

Each tutorial carries its own `Allrun`, which meshes the case and runs the
solver. The equivalent by hand, which is worth doing once to watch the output
scroll past:

```bash
blockMesh
rm -rf 0 && cp -r 0.orig 0   # fields ship in 0.orig; every Allrun does this
coupledSimpleFoam
```

The repository root has an `Allrun` that runs every tutorial - `cavity`,
`stripChannel`, `finnedDuct` and `stripChannelRAS` - and prints a pass/fail
heatedDuct        the heated duct, energy on
summary, so it doubles as a regression gate:

```bash
cd /path/to/coupledSimpleFoam
./Allrun
```

A case counts as failed if its `Allrun` exits non-zero or if `FOAM FATAL ERROR`
appears in any of its logs; the root script exits non-zero if any case failed.
Note that `finnedDuct` is three-dimensional and `stripChannelRAS` is RANS, so
both take noticeably longer than `cavity`.

### What a successful log looks like

Start-up: the solver announces the system size, the backend it selected, and
how the pressure level is set. The cavity has no `fixedValue` pressure patch,
so the pressure is pinned:

```
Pressure level: pinned at global cell 0 to p = 0

Coupled p-U system: 16384 unknowns, backend EigenSparseLU

Starting coupled iteration loop
```

16384 = 4 unknowns per cell x 4096 cells, which is the first thing to check if
a case behaves oddly.

The first outer iteration:

```
Time = 1s

EigenSparseLU:  Solving for Ux, Initial residual = 1, Final residual = 2.00854400236e-15, No Iterations 1
EigenSparseLU:  Solving for Uy, Initial residual = 1, Final residual = 0.747110880419, No Iterations 1
EigenSparseLU:  Solving for Uz, Initial residual = 0, Final residual = 0, No Iterations 1
EigenSparseLU:  Solving for p, Initial residual = 0, Final residual = 1.77009198474e-14, No Iterations 1
time step continuity errors : sum local = 1.28080978574e-16, global = -8.98498470606e-19, cumulative = -8.98498470606e-19
coupled assembly check: sum|div(phi)| = 1.28080978574e-16 (excl. pRef cell 1.2643671557e-16), max|div(phi)| = 1.64426300419e-18, normalised = 1.75162884398e-16
```

Four things to read here:

- The residual lines keep the standard OpenFOAM format, so `foamLog` and
  PyFoam parse them unchanged. `No Iterations 1` is correct for a direct
  solve.
- `sum|div(phi)|` is at machine precision, ~1e-16. This is the assembly
  self-check: the discrete continuity rows are satisfied exactly because the
  flux is rebuilt from the same expression that went into the matrix. Any
  case, on any mesh, should show this from the first iteration. A value that
  is not near machine zero means the assembly is wrong for that case, not that
  the case has not converged yet.
- `Uz` is identically zero on a 2-D mesh: empty directions get an identity
  momentum row.
- The `Uy` *final* residual of 0.747 at the first iteration is a known
  normalisation artefact, not a failure. For a block whose reference response
  is numerically zero - which `Uy` is at the first iteration of a cavity -
  the OpenFOAM-style normalisation factor collapses. The absolute residual is
  ~1e-15, as `Ux` shows on the same line. Stock OpenFOAM shows the same
  artefact for near-zero fields, and convergence control uses only the
  *initial* residuals.

### Expected iteration count

The run **converges in 12 outer iterations** on the 64 x 64 cavity, and ends:

```
EigenSparseLU:  Solving for Ux, Initial residual = 5.30956157716e-08, ...
EigenSparseLU:  Solving for Uy, Initial residual = 3.81722024593e-06, ...
EigenSparseLU:  Solving for Uz, Initial residual = 0, ...
EigenSparseLU:  Solving for p, Initial residual = 8.14870195515e-07, ...

COUPLED solution converged in 12 iterations

ExecutionTime = ... s  ClockTime = ... s
End
```

Wall clock is well under a second on a current desktop CPU. For orientation,
the segregated reference solver (`foamRun -solver incompressibleFluid`, SIMPLE
with relaxation 0.7 / 0.3) needs several hundred outer iterations on the same
mesh at the same `residualControl`.

That 12 is a strong check, because it is a property of the discretisation, not
of the machine: it is identical in serial and in parallel, and identical
between the Eigen and PETSc backends. If the tutorial converges in 12
iterations, the assembly, the boundary-condition handling and the pressure
pinning are all working.

### Verify the parallel build

Only meaningful in a PETSc build. `numberOfSubdomains` in
`system/decomposeParDict` must match the `-np` argument, so check it - or write
one - before decomposing:

```bash
cd /path/to/coupledSimpleFoam
cp -r tutorials/cavity /tmp/cavity-parallel
cd /tmp/cavity-parallel

grep numberOfSubdomains system/decomposeParDict     # must equal the -np below

# cavity pins the backend to the serial Eigen solver; drop that line so the
# parallel PETSc backend is selected
sed -i '/linearSolver *SparseLU/d' system/fvSolution

blockMesh
rm -rf 0 && cp -r 0.orig 0
decomposePar -force
mpirun -np 4 coupledSimpleFoam -parallel < /dev/null
reconstructPar
```

The start-up banner should now read `backend PETSc`, and the run should take
the **same 12 outer iterations** - identical, not within one. Rank count does
not change the outer iteration count, because the parallel backend solves the
same assembled system; if it does change, the decomposition or the
processor-patch coupling is at fault, not the tolerance.

To confirm the parallel answer against the serial one field by field, keep a
serial run of the same case alongside it and compare the two:

```bash
cd /path/to/coupledSimpleFoam
cp -r tutorials/cavity /tmp/cavity-serial
( cd /tmp/cavity-serial && blockMesh && cp -r 0.orig 0 && coupledSimpleFoam )

# usage: compare_fields.py <caseA> <timeA> <caseB> <timeB> [meshSize]
python3 tools/compare_fields.py /tmp/cavity-parallel 12 /tmp/cavity-serial 12 64
```

Both runs converge at iteration 12, so `12` is the written time directory in
each. Serial and parallel agree to about 1e-10 or better; the remaining
difference is dominated by the ASCII `writePrecision` in `controlDict`, not by
the linear algebra.

---

## 8. Troubleshooting

| Symptom | Cause | Fix |
| --- | --- | --- |
| `Allwmake: the OpenFOAM environment is not set`, or `wmake: command not found` | `etc/bashrc` was never sourced in this shell. Non-interactive shells do not read `~/.bashrc`. | `. /opt/openfoam14/etc/bashrc`, then rerun. Add it to `~/.bashrc` for interactive shells and source it explicitly at the top of any script. |
| `fatal error: Eigen/Sparse: No such file or directory` | Eigen headers absent, or not in `/usr/include/eigen3`. `Allwmake` warns about this before compiling. | `sudo apt-get install -y libeigen3-dev`. If Eigen is elsewhere, `EIGEN_INC=-I/path/to/eigen ./Allwmake`. |
| `Allwmake` prints `PETSc backend : not found by pkg-config`, and parallel runs then fail | `petsc-dev` is not installed, or its `.pc` file is not on the `pkg-config` search path. | `sudo apt-get install -y petsc-dev`. Check with `pkg-config --exists PETSc; echo $?` (0 = found). For a non-standard location: `export PKG_CONFIG_PATH=/path/to/petsc/lib/pkgconfig:$PKG_CONFIG_PATH`, then rebuild. Verify with `pkg-config --libs PETSc`. |
| `linearSolver PETSc requested but coupledSimpleFoam was built without PETSc support` | The binary in `$FOAM_USER_APPBIN` is an Eigen-only build - either `COUPLED_NO_PETSC=1` was set, or PETSc was not found when it was compiled. | Install PETSc, then `./Allclean && ./Allwmake` and re-read the `PETSc backend` line. Confirm with `ldd "$FOAM_USER_APPBIN/coupledSimpleFoam" \| grep petsc`. |
| `linearSolver SparseLU is a serial (Eigen) backend but this is a parallel run` | An Eigen backend was requested with `mpirun`. Eigen is serial only, by construction. | Set `linearSolver PETSc;` in the `COUPLED` dictionary, or drop the entry - it defaults to PETSc in a parallel run and SparseLU in serial. |
| `Unknown Eigen backend <name>. Valid options are SparseLU and BiCGSTAB` | Typo in `linearSolver`. Values are case-sensitive. | Use exactly `SparseLU`, `BiCGSTAB` or `PETSc`. |
| Missing MPI symbols at link, a crash inside `MPI_Init`, "MPI_Init invoked multiple times", or `mpirun -np 4` in which every rank reports rank 0 of 1 | PETSc and OpenFOAM link different MPI implementations or different builds of one. | Compare `mpicc --showme:link` against `pkg-config --libs PETSc \| tr ' ' '\n' \| grep -i mpi`, and check `ldd "$FOAM_USER_APPBIN/coupledSimpleFoam" \| grep -i mpi` shows exactly one `libmpi`. Rebuild PETSc against the system MPI (section 4.2) or use the distribution PETSc, which already matches. |
| `mpicc --showme:compile` returns nothing and PETSc headers are then not found | The MPI stack is MPICH; `--showme` is an OpenMPI extension. | Add the MPI include and library flags by hand through `EXE_INC` / `EXE_LIBS` in `Make/options`, or put an `mpicc` wrapper that accepts `--showme` on `PATH`. |
| `coupledSimpleFoam: command not found`, or "no such solver" when a script calls it | The environment is not sourced (so `$FOAM_USER_APPBIN` is not on `PATH`), or the build has not been run, or the case is being run through `foamRun -solver ...`. | `. /opt/openfoam14/etc/bashrc`; check `ls -l "$FOAM_USER_APPBIN/coupledSimpleFoam"` and `command -v coupledSimpleFoam`. `coupledSimpleFoam` is a standalone application, invoked by name - it is **not** a `foamRun` solver module and cannot be selected with `-solver`. |
| `cannot find file .../constant/momentumTransport` at start-up | Every case needs `constant/momentumTransport`, laminar ones included. The framework falls back to `constant/turbulenceProperties`; if neither exists the run dies in the dictionary read. | Add `constant/momentumTransport` containing `simulationType laminar;` (or `simulationType RAS;` plus a `RAS { model ...; turbulence on; }` block). Also required: `constant/physicalProperties` with a `viscosityModel` entry. Copy the layout from any tutorial. |
| `keyword wallDist is undefined in dictionary system/fvSchemes` with `kOmegaSST` | Wall-distance-based models need a `wallDist` method; the schemes dictionary has no default. | Add to `system/fvSchemes`: `wallDist { method meshWave; }`. |
| `Eigen SparseLU factorisation failed (singular matrix?)`, or a PETSc `KSP` that diverges immediately with no `fixedValue` pressure patch anywhere | The pressure level is undetermined. In a closed domain (or one with only fixed-flux boundaries) the pressure is defined up to a constant, and the system is singular unless a reference is pinned. | Nothing is needed in the usual case - the solver detects that no `p` patch reports `fixesValue()` and pins the continuity row of `pRefCell` to `pRefValue`. Check the start-up line: `Pressure level: pinned at global cell 0 to p = 0` versus `Pressure level: set by fixedValue patch <name>`. If pinning is happening but the matrix is still singular, `pRefCell` may be out of range - it is a **global** cell index, not a per-rank one - so set a valid `pRefCell` in the `COUPLED` dictionary. Also check that inflow and outflow balance: no `adjustPhi`-style flux scaling is applied. |
| `Make/options:NN: *** missing separator. Stop.`, `/bin/sh^M: bad interpreter`, or dictionary parse errors on files that look fine | CRLF line endings, from editing or checking out on Windows. `Make/options` is read raw by GNU make, and the shell scripts are `sh`. | The repository ships a `.gitattributes` that forces LF; make sure it is honoured (`git config core.autocrlf input`, then re-checkout). To repair an existing tree: `sudo apt-get install -y dos2unix && find . -type f \( -name '*.C' -o -name '*.H' -o -name 'files' -o -name 'options' -o -name 'All*' \) -exec dos2unix -q {} +`. Under WSL, editing with a Windows editor is the usual source. (A C-style `/* */` comment in `Make/options` produces the same `missing separator` error - use `#`.) |
| `decomposePar` refuses to run, or a parallel run reads the wrong mesh, or rank counts and `numberOfSubdomains` disagree | Stale `processor*` directories from an earlier decomposition, often with a different rank count. | `rm -rf processor*` before `decomposePar`, or run `decomposePar -force`. `./Allclean` does this for every tutorial. Always re-run `blockMesh` and `decomposePar` after changing `numberOfSubdomains`. |
| `COUPLED did NOT converge within N outer iterations` on a case that should converge | Most often the inner linear tolerance is too loose. The outer loop is a Picard iteration with a nonzero initial guess, so a loose `relTol` makes each linear update negligible and the outer loop stalls rather than diverges. | Tighten `relTol` in the `COUPLED` dictionary (1e-12 for small laminar cases). `doc/USAGE.md` has the per-regime table - the correct value differs for large 3-D laminar and for RANS, and looser is not always faster. Raising `nOuterIterations` alone does not help a stalled loop. |
| PETSc `KSP` reports `DIVERGED_ITS` at the inner iteration cap every outer iteration | The preconditioner does not suit the mesh topology. No single preconditioner is robust everywhere. | The shipped default (fieldsplit-Schur with hypre BoomerAMG on the Schur block) is the only configuration measured to work on every topology tested - first confirm it is actually in use and that PETSc has hypre. `doc/USAGE.md` lists the alternatives; all are one `petscOptions` line and need no rebuild. |
| `-log_view` in `petscOptions` has no effect | It is consumed inside `PetscInitialize`, which runs before the dictionary string is inserted. | Pass it through the environment: `export PETSC_OPTIONS="-log_view"` and `mpirun -np 6 -x PETSC_OPTIONS coupledSimpleFoam -parallel < /dev/null`. |
| A script piped into a shell stops silently after the first `mpirun` | `mpirun` drains standard input. | Give every `mpirun` a `< /dev/null`. |
| Compiling is extremely slow under WSL | The working tree is on `/mnt/c`. | Move the checkout into the Linux filesystem (section 5.1). |

---

## Appendix: from nothing to a verified build

### A. Native Ubuntu 22.04 / 24.04, CPU

```bash
# --- [one-off] ------------------------------------------------------------
sudo apt-get update
sudo apt-get install -y wget software-properties-common git
sudo sh -c "wget -O - https://dl.openfoam.org/gpg.key > /etc/apt/trusted.gpg.d/openfoam.asc"
sudo add-apt-repository "http://dl.openfoam.org/ubuntu main dev"
sudo apt-get update
sudo apt-get install -y openfoam14 libeigen3-dev petsc-dev

echo '. /opt/openfoam14/etc/bashrc' >> ~/.bashrc

# --- [per-shell] ----------------------------------------------------------
. /opt/openfoam14/etc/bashrc

# --- [one-off] ------------------------------------------------------------
git clone https://github.com/dhanninacs/coupledSimpleFoam.git "$HOME/coupledSimpleFoam"
cd "$HOME/coupledSimpleFoam"
./Allwmake

# --- verify ---------------------------------------------------------------
command -v coupledSimpleFoam
cd tutorials/cavity && blockMesh && cp -r 0.orig 0 && coupledSimpleFoam
#   -> COUPLED solution converged in 12 iterations
```

### B. Windows, WSL2 Ubuntu, CPU

Identical to A, run inside the WSL distribution, with one requirement: keep the
checkout in the Linux filesystem (`$HOME/...`), not under `/mnt/c`.

```powershell
# In Windows PowerShell, one-off
wsl --install -d Ubuntu-24.04
```

Then open the Ubuntu shell and follow recipe A verbatim.

### C. Other Linux distributions

There are no Foundation packages outside Debian/Ubuntu, so OpenFOAM 14 has to
be built from source (or run from a container image published by
openfoam.org). Everything after that is unchanged; only the dependency package
names differ:

| Distribution | Eigen | PETSc | Notes |
| --- | --- | --- | --- |
| Fedora / RHEL | `eigen3-devel` | `petsc-devel` | MPI comes from an environment module: `module load mpi/openmpi-x86_64` before building or running. |
| openSUSE | `eigen3-devel` | `petsc-devel` | |
| Arch | `eigen` | `petsc` (AUR) | Headers may land in `/usr/include/eigen3`; if not, set `EIGEN_INC`. |

Two portability points:

- The `pkg-config` module is named `PETSc` on Debian/Ubuntu and `petsc` on most
  source builds. `Make/options` tries both, so either works; if neither is
  found, set `PKG_CONFIG_PATH` to the directory holding the `.pc` file.
- If the Eigen headers are not under `/usr/include/eigen3`, pass their location
  with `EIGEN_INC=-I/path/to/eigen`.

---

## Where to go next

- [`doc/USAGE.md`](USAGE.md) - the `COUPLED` dictionary, every key and its
  default, preconditioner selection, the per-regime tolerance guidance, and the
  full list of limitations.
- [`doc/BENCHMARKS.md`](BENCHMARKS.md) - measured performance against segregated
  SIMPLE, MPI scaling, and the preconditioner study.
- [`doc/VALIDATION.md`](VALIDATION.md) - verification evidence and the
  acceptance criteria behind the numbers quoted in section 7.
- [`doc/THEORY.md`](THEORY.md) - the coupled formulation itself.
- [`doc/DEVELOPMENT.md`](DEVELOPMENT.md) - code layout, the OpenFOAM 14 API
  notes the implementation depends on, and how to add a backend.
- [`CONTRIBUTING.md`](../CONTRIBUTING.md) - how to build, test and submit a
  change.
