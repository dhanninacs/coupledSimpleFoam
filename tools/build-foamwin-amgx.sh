#!/usr/bin/env bash
# Cross-build the AmgX coupledSimpleFoam for FoamWin (MinGW), as FoamWin's scripts/build-coupled.sh
# builds the serial one: sources from the feat/amgx-backend checkout, no PETSc, AmgX through the
# Windows amgxsh.dll (MSVC + CUDA, C API only) and its import library. Installs
# FOAM_APPBIN/amgx/coupledSimpleFoam.exe in the WSL FoamWin tree, with amgxsh.dll beside it.
set -eo pipefail
SRC="$(cd "$(dirname "$0")/.." && pwd)/src/coupledSimpleFoam"
ROOT="$HOME/winbuild/csf-amgx-win"
BUILD="$ROOT/coupledSimpleFoam"
AMGXW="$ROOT/amgx"
WINDLL="${AMGX_WINDOWS_BUILD:?set AMGX_WINDOWS_BUILD to the folder with amgxsh.dll and amgxsh.lib}"
rm -rf "$BUILD"
mkdir -p "$BUILD/Make" "$AMGXW/lib" "$AMGXW/include"
cp "$SRC"/*.[CH] "$BUILD/"
cp "$SRC/Make/files" "$SRC/Make/options" "$BUILD/Make/"
cp "$HOME"/amgx-src/include/*.h "$AMGXW/include/"
cp "$WINDLL/amgxsh.dll" "$WINDLL/amgxsh.lib" "$AMGXW/lib/"
sed -i 's#^EXE *=.*#EXE = $(FOAM_USER_APPBIN)/coupledSimpleFoam#' "$BUILD/Make/files"

cd "$HOME/winbuild/foamWin/OpenFOAM-14"
set +u
source etc/pre-bashrc-mingw_w64.sh
export EIGEN_INC="-I/usr/include/eigen3" COUPLED_NO_PETSC=1 AMGX_DIR="$AMGXW" AMGX_RPATH=
export FOAM_USER_APPBIN="$FOAM_APPBIN/amgx"
mkdir -p "$FOAM_USER_APPBIN"
(cd "$BUILD" && wmake -j "${JOBS:-8}") > "$ROOT/build.log" 2>&1 || { grep -E 'error|Error' "$ROOT/build.log" | head -20; exit 1; }
cp "$AMGXW/lib/amgxsh.dll" "$FOAM_USER_APPBIN/"
ls -l "$FOAM_USER_APPBIN"
x86_64-w64-mingw32-objdump -p "$FOAM_USER_APPBIN/coupledSimpleFoam.exe" | grep "DLL Name"
echo "FOAM_USER_APPBIN=$FOAM_USER_APPBIN"
