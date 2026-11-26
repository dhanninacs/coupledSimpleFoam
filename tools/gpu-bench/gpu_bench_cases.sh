#!/usr/bin/env bash
# Make the public GPU benchmark cases for coupledSimpleFoam in ~/gpu-bench.
#   mkcase <name> cavity|duct <n> <energy yes|no> <device cpu|gpu> [np]
# cavity: unit cube, lid (1 0 0) m/s, Re 400 (nu 0.0025), n^3 cells. Reference: 3-D lid-driven cavity.
# duct:   square duct 1 x 1 x 8, Re 20 (U 1, Dh 1, nu 0.05), n x n x 8n cells; walls 350 K, inlet 300 K,
#         Pr 0.7. Reference: fully developed laminar Nu_T = 2.98 for a square duct (Shah & London).
# The COUPLED strings are coupledSimpleFoam's own: CPU from doc/BENCHMARKS.md 9.2, device from
# coupledSimpleFoam-gpu doc/USAGE-GPU.md section 1.
set -euo pipefail
root="$HOME/gpu-bench"
mkdir -p "$root"

header() { printf 'FoamFile\n{\n    format ascii;\n    class %s;\n    object %s;\n}\n' "$1" "$2"; }

CPU_OPTS='-ksp_type fbcgs -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper -pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type boomeramg -fieldsplit_1_pc_hypre_boomeramg_strong_threshold 0.5 -fieldsplit_1_pc_hypre_boomeramg_coarsen_type HMIS -fieldsplit_1_pc_hypre_boomeramg_interp_type ext+i -fieldsplit_1_pc_hypre_boomeramg_P_max 4 -fieldsplit_1_pc_hypre_boomeramg_truncfactor 0.3 -fieldsplit_1_pc_hypre_boomeramg_agg_nl 1 -fieldsplit_1_pc_hypre_boomeramg_relax_type_all SOR/Jacobi'
GPU_OPTS='-mat_type aijcusparse -vec_type cuda -ksp_type fbcgs -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper -pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_block_size 4 -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 -fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type pbjacobi -fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type boomeramg -fieldsplit_1_pc_hypre_boomeramg_strong_threshold 0.5 -fieldsplit_1_pc_hypre_boomeramg_coarsen_type PMIS -fieldsplit_1_pc_hypre_boomeramg_interp_type ext+i -fieldsplit_1_pc_hypre_boomeramg_P_max 4 -fieldsplit_1_pc_hypre_boomeramg_truncfactor 0.3 -fieldsplit_1_pc_hypre_boomeramg_agg_nl 1 -fieldsplit_1_pc_hypre_boomeramg_relax_type_all l1scaled-Jacobi -fieldsplit_1_pc_hypre_boomeramg_no_CF'

mkcase() {
  local name=$1 kind=$2 n=$3 energy=$4 device=$5 np=${6:-1}
  local d="$root/$name"
  rm -rf "$d"; mkdir -p "$d/system" "$d/constant" "$d/0"
  local nu opts walls
  if [ "$kind" = cavity ]; then nu=0.0025; else nu=0.05; fi
  if [ "$device" = gpu ]; then opts="$GPU_OPTS"; else opts="$CPU_OPTS"; fi
  local backend=PETSc extra=""
  if [ "$device" = amgx ]; then backend=AmgX; extra="amgxPrecision ${AMGXP:-double}; amgxReuse ${AMGXREUSE:-resetup};"; fi
  [ -n "${AMGXCFG:-}" ] && extra="$extra amgxConfig \"$AMGXCFG\";"

  { header dictionary controlDict
    cat <<EOF
application coupledSimpleFoam;
solver incompressibleFluid;
startFrom startTime; startTime 0; stopAt endTime; endTime ${NOUTER:-80}; deltaT 1;
writeControl timeStep; writeInterval 1000; purgeWrite 1;
writeFormat binary; writePrecision 12; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable false;
EOF
  } > "$d/system/controlDict"

  { header dictionary fvSchemes
    cat <<'EOF'
ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div(phi,T) bounded Gauss linearUpwind grad(T); div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
EOF
  } > "$d/system/fvSchemes"

  { header dictionary fvSolution
    cat <<EOF
solvers
{
    p { solver GAMG; smoother GaussSeidel; tolerance 1e-08; relTol 0.01; }
    U { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-07; relTol 0.1; }
    T { solver ${TSOLVER:-PBiCGStab}; preconditioner DILU; tolerance 1e-12; relTol 0; maxIter 1000; }
}
SIMPLE { nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; residualControl { p 1e-4; U 1e-5; } }
relaxationFactors { equations { U 0.7; } fields { p 0.3; } }
COUPLED
{
    nOuterIterations    ${NOUTER:-80};
    linearSolver        $backend;
    $extra
    pRefCell            0;
    pRefValue           0;
    pseudoTransient     ${PSEUDO:-yes};
    pseudoCo            ${PCO:-50};
    adaptiveTolerance   yes;
    relTolMax           3e-2;
    relTolMin           1e-8;
    adaptiveGamma       0.9;
    adaptiveAlpha       1.5;
    pcReuseInterval     ${REUSE:-8};
    maxLinearIter       ${MAXLIN:-500};
    solveEnergy         $energy;
    Pr                  0.7;
    petscOptions        "$opts";
    residualControl     { U 1e-5; p 1e-4; T 1e-6; }
}
EOF
  } > "$d/system/fvSolution"

  { header dictionary decomposeParDict
    printf 'numberOfSubdomains %s;\nmethod scotch;\n' "$np"
  } > "$d/system/decomposeParDict"

  { header dictionary physicalProperties; printf 'viscosityModel constant;\nnu %s;\n' "$nu"; } > "$d/constant/physicalProperties"
  { header dictionary momentumTransport; printf 'simulationType laminar;\n'; } > "$d/constant/momentumTransport"

  if [ "$kind" = cavity ]; then
    { header dictionary blockMeshDict
      cat <<EOF
scale 1;
vertices ((0 0 0) (1 0 0) (1 1 0) (0 1 0) (0 0 1) (1 0 1) (1 1 1) (0 1 1));
blocks (hex (0 1 2 3 4 5 6 7) ($n $n $n) simpleGrading (1 1 1));
boundary
(
    movingWall { type wall; faces ((3 7 6 2)); }
    fixedWalls { type wall; faces ((0 4 7 3) (2 6 5 1) (1 5 4 0) (0 3 2 1) (4 5 6 7)); }
);
EOF
    } > "$d/system/blockMeshDict"
    { header volVectorField U
      printf 'dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{\n    movingWall { type fixedValue; value uniform (1 0 0); }\n    fixedWalls { type noSlip; }\n}\n'
    } > "$d/0/U"
    { header volScalarField p
      printf 'dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n    movingWall { type zeroGradient; }\n    fixedWalls { type zeroGradient; }\n}\n'
    } > "$d/0/p"
  else
    local nx=$((8 * n))
    { header dictionary blockMeshDict
      cat <<EOF
scale 1;
vertices ((0 0 0) (8 0 0) (8 1 0) (0 1 0) (0 0 1) (8 0 1) (8 1 1) (0 1 1));
blocks (hex (0 1 2 3 4 5 6 7) ($nx $n $n) simpleGrading (1 1 1));
boundary
(
    inlet  { type patch; faces ((0 4 7 3)); }
    outlet { type patch; faces ((2 6 5 1)); }
    walls  { type wall;  faces ((1 5 4 0) (3 7 6 2) (0 3 2 1) (4 5 6 7)); }
);
EOF
    } > "$d/system/blockMeshDict"
    { header volVectorField U
      printf 'dimensions [0 1 -1 0 0 0 0];\ninternalField uniform (0 0 0);\nboundaryField\n{\n    inlet { type fixedValue; value uniform (1 0 0); }\n    outlet { type zeroGradient; }\n    walls { type noSlip; }\n}\n'
    } > "$d/0/U"
    { header volScalarField p
      printf 'dimensions [0 2 -2 0 0 0 0];\ninternalField uniform 0;\nboundaryField\n{\n    inlet { type zeroGradient; }\n    outlet { type fixedValue; value uniform 0; }\n    walls { type zeroGradient; }\n}\n'
    } > "$d/0/p"
    { header volScalarField T
      printf 'dimensions [0 0 0 1 0 0 0];\ninternalField uniform 300;\nboundaryField\n{\n    inlet { type fixedValue; value uniform 300; }\n    outlet { type zeroGradient; }\n    walls { type fixedValue; value uniform 350; }\n}\n'
    } > "$d/0/T"
  fi
  echo "made $d ($kind n=$n energy=$energy device=$device np=$np)"
}

"$@"
