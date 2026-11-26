"""The heated square duct for Stokes's default heated engine, `foamRun -solver fluid` (constant
density, as Stokes's of14.fluid writer does for a liquid): the mesh of an existing duct case, walls
at 350 K, inlet 300 K, Re 20, Pr 0.7, rho 1, Cv 1e5 (so pressure work and kinetic energy are ~1e-4 K
and the temperature is the passive scalar the coupled solver solves).

usage: python3 heated_fluid_case.py <duct case with constant/polyMesh> <new case> [np]
"""

import pathlib
import shutil
import sys

src, dst = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2])
np_ = int(sys.argv[3]) if len(sys.argv) > 3 else 10
if dst.exists():
    shutil.rmtree(dst)
shutil.copytree(src / "constant" / "polyMesh", dst / "constant" / "polyMesh")


def header(cls: str, obj: str) -> str:
    return f"FoamFile\n{{\n    format ascii;\n    class {cls};\n    object {obj};\n}}\n\n"


def write(rel: str, cls: str, obj: str, body: str) -> None:
    p = dst / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header(cls, obj) + body)


write("system/controlDict", "dictionary", "controlDict", """application foamRun;
solver fluid;
startFrom startTime; startTime 0; stopAt endTime; endTime 4000; deltaT 1;
writeControl timeStep; writeInterval 4000; purgeWrite 1;
writeFormat binary; writePrecision 12; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable false;
""")
write("system/decomposeParDict", "dictionary", "decomposeParDict", f"numberOfSubdomains {np_};\nmethod scotch;\n")
write("system/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
gradSchemes { default Gauss linear; }
divSchemes
{
    default none;
    div(phi,U) bounded Gauss linearUpwind grad(U);
    div(phi,e) bounded Gauss linearUpwind grad(e);
    div(phi,K) bounded Gauss linear;
    div(phi,(p|rho)) Gauss linear;
    div(((rho*nuEff)*dev2(T(grad(U))))) Gauss linear;
}
laplacianSchemes { default Gauss linear corrected; }
interpolationSchemes { default linear; }
snGradSchemes { default corrected; }
""")
write("system/fvSolution", "dictionary", "fvSolution", """solvers
{
    "rho.*" { solver diagonal; }
    p_rgh { solver GAMG; smoother GaussSeidel; tolerance 1e-08; relTol 0.01; }
    "(U|e)" { solver smoothSolver; smoother symGaussSeidel; tolerance 1e-09; relTol 0.1; }
}
PIMPLE
{
    momentumPredictor yes;
    nNonOrthogonalCorrectors 0;
    residualControl { p_rgh 1e-4; U 1e-5; e 1e-6; }
}
relaxationFactors { fields { p_rgh 0.3; } equations { U 0.7; e 0.9; } }
""")
write("constant/physicalProperties", "dictionary", "physicalProperties", """thermoType
{
    type heRhoThermo; mixture pureMixture; transport const; thermo eConst;
    equationOfState rhoConst; specie specie; energy sensibleInternalEnergy;
}
mixture
{
    specie { molWeight 18; }
    equationOfState { rho 1; }
    thermodynamics { Cv 1e5; hf 0; }
    transport { mu 0.05; Pr 0.7; }
}
""")
write("constant/momentumTransport", "dictionary", "momentumTransport", "simulationType laminar;\n")
write("constant/g", "uniformDimensionedVectorField", "g", "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 0);\n")
write("constant/pRef", "uniformDimensionedScalarField", "pRef", "dimensions [1 -1 -2 0 0 0 0];\nvalue 0;\n")
write("0/U", "volVectorField", "U", """dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{
    inlet { type fixedValue; value uniform (1 0 0); }
    outlet { type zeroGradient; }
    walls { type noSlip; }
}
""")
write("0/p_rgh", "volScalarField", "p_rgh", """dimensions [1 -1 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type fixedFluxPressure; value uniform 0; }
    outlet { type fixedValue; value uniform 0; }
    walls { type fixedFluxPressure; value uniform 0; }
}
""")
write("0/p", "volScalarField", "p", """dimensions [1 -1 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{
    inlet { type calculated; value uniform 0; }
    outlet { type calculated; value uniform 0; }
    walls { type calculated; value uniform 0; }
}
""")
write("0/T", "volScalarField", "T", """dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{
    inlet { type fixedValue; value uniform 300; }
    outlet { type zeroGradient; }
    walls { type fixedValue; value uniform 350; }
}
""")
print(f"made {dst}")
