"""Turn a split conjugate-duct case (cht_case.py + splitMeshRegions) into the same case for
coupledSimpleFoam's conjugate mode: same meshes, the fluid in coupledSimpleFoam's own format
(kinematic p, constant nu), the interfaces fixedValue (the conjugate solve sets them).

usage: python3 cht_to_coupled.py <split case> <new case> <coupled: AmgX|PETSc> <conjugate: AmgX|Eigen>
       [nOuter]
"""

import pathlib
import shutil
import sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
coupled, conj = sys.argv[3], sys.argv[4]
nouter = int(sys.argv[5]) if len(sys.argv) > 5 else 200

if dst.exists():
    shutil.rmtree(dst)
for region in ("fluid", "solid"):
    shutil.copytree(src / "constant" / region / "polyMesh", dst / "constant" / region / "polyMesh")
shutil.copy(src / "constant" / "solid" / "physicalProperties", dst / "constant" / "solid" / "physicalProperties")
info = (src / "case.info").read_text()
rhocp = dict(kv.split("=") for kv in info.split())["rhoCp"]
(dst / "case.info").write_text(info)


def header(cls: str, obj: str) -> str:
    return f"FoamFile\n{{\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n\n"


def write(rel: str, cls: str, obj: str, body: str) -> None:
    p = dst / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header(cls, obj) + body)


CPU_OPTS = ("-ksp_type fbcgs -pc_type fieldsplit -pc_fieldsplit_type schur -pc_fieldsplit_schur_fact_type upper "
            "-pc_fieldsplit_schur_precondition a11 -pc_fieldsplit_0_fields 0,1,2 -pc_fieldsplit_1_fields 3 "
            "-fieldsplit_0_ksp_type preonly -fieldsplit_0_pc_type bjacobi -fieldsplit_0_sub_pc_type ilu "
            "-fieldsplit_1_ksp_type preonly -fieldsplit_1_pc_type hypre -fieldsplit_1_pc_hypre_type boomeramg "
            "-fieldsplit_1_pc_hypre_boomeramg_strong_threshold 0.5 -fieldsplit_1_pc_hypre_boomeramg_coarsen_type HMIS "
            "-fieldsplit_1_pc_hypre_boomeramg_interp_type ext+i -fieldsplit_1_pc_hypre_boomeramg_P_max 4 "
            "-fieldsplit_1_pc_hypre_boomeramg_truncfactor 0.3 -fieldsplit_1_pc_hypre_boomeramg_agg_nl 1 "
            "-fieldsplit_1_pc_hypre_boomeramg_relax_type_all SOR/Jacobi")
backend = f'linearSolver AmgX; amgxReuse resetup;' if coupled == "AmgX" else f'linearSolver PETSc; petscOptions "{CPU_OPTS}";'

write("system/controlDict", "dictionary", "controlDict", f"""application coupledSimpleFoam;
startFrom startTime; startTime 0; stopAt endTime; endTime {nouter}; deltaT 1;
writeControl timeStep; writeInterval 100000; purgeWrite 1;
writeFormat binary; writePrecision 12; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable false;
""")
for region in ("fluid", "solid"):
    write(f"system/{region}/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes { default none; div(phi,U) bounded Gauss linearUpwind grad(U); div(phi,T) bounded Gauss linearUpwind grad(T); div((nuEff*dev2(T(grad(U))))) Gauss linear; }
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
write("system/solid/fvSolution", "dictionary", "fvSolution", "solvers {}\n")
write("system/fluid/fvSolution", "dictionary", "fvSolution", f"""solvers
{{
    p {{ solver GAMG; smoother GaussSeidel; tolerance 1e-08; relTol 0.01; }}
    U {{ solver smoothSolver; smoother symGaussSeidel; tolerance 1e-07; relTol 0.1; }}
}}
SIMPLE {{ nNonOrthogonalCorrectors 0; pRefCell 0; pRefValue 0; }}
COUPLED
{{
    nOuterIterations    {nouter};
    {backend}
    pseudoTransient     yes;
    pseudoCo            50;
    adaptiveTolerance   yes;
    relTolMax           3e-2;
    relTolMin           1e-8;
    adaptiveGamma       0.9;
    adaptiveAlpha       1.5;
    pcReuseInterval     8;
    maxLinearIter       500;
    solveEnergy         yes;
    Pr                  0.7;
    solidRegions        (solid);
    rhoCp               {rhocp};
    conjugateSolver     {conj};
    conjugateTolerance  1e-9;
    conjugateMaxIter    2000;
    timing              yes;
    residualControl     {{ U 1e-5; p 1e-4; T 1e-6; }}
}}
""")
write("constant/fluid/physicalProperties", "dictionary", "physicalProperties", "viscosityModel constant;\nnu 0.05;\n")
write("constant/fluid/momentumTransport", "dictionary", "momentumTransport", "simulationType laminar;\n")
iface = "fluid_to_solid"
write("0/fluid/U", "volVectorField", "U", f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform (1 0 0); }}
    outlet {{ type zeroGradient; }}
    {iface} {{ type noSlip; }}
}}
""")
write("0/fluid/p", "volScalarField", "p", f"""dimensions [0 2 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type zeroGradient; }}
    outlet {{ type fixedValue; value uniform 0; }}
    {iface} {{ type zeroGradient; }}
}}
""")
write("0/fluid/T", "volScalarField", "T", f"""dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{{
    inlet {{ type fixedValue; value uniform 300; }}
    outlet {{ type zeroGradient; }}
    {iface} {{ type fixedValue; value uniform 300; }}
}}
""")
write("0/solid/T", "volScalarField", "T", """dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{
    solidEnds { type zeroGradient; }
    outerWall { type fixedValue; value uniform 350; }
    solid_to_fluid { type fixedValue; value uniform 300; }
}
""")
print(f"made {dst} coupled={coupled} conjugate={conj}")
