"""A public conjugate-heat-transfer benchmark for OpenFOAM 14's foamMultiRun (and later the GPU solver):
a square fluid duct 1 x 1 x 8 (Re 20, Pr 0.7) inside a solid shell 0.25 thick whose outer surface is
held at 350 K; the fluid enters at 300 K. Regions "fluid" and "solid" from cell zones
(splitMeshRegions -cellZones), coupled by coupledTemperature on mappedWall interfaces, steady.

usage: python3 cht_case.py <case dir> <n: cells across the duct> [m: cells across the shell]
"""

import pathlib
import sys

case = pathlib.Path(sys.argv[1])
n = int(sys.argv[2])
m = int(sys.argv[3]) if len(sys.argv) > 3 else max(2, n // 4)
L, t = 8.0, 0.25
nx = 8 * n

RHO, CP, MU, PR = 1.0, 1.0e5, 0.05, 0.7  # Cp large: pressure work and kinetic energy (in the
# compressible fluid module's energy equation) become ~1e-4 K, so the reference is a passive scalar.
K_FLUID = RHO * CP * (MU / RHO) / PR
K_SOLID = 10.0 * K_FLUID


def header(cls: str, obj: str, location: str = "") -> str:
    loc = f'    location "{location}";\n' if location else ""
    return f"FoamFile\n{{\n    format ascii;\n    class {cls};\n{loc}    object {obj};\n}}\n\n"


def write(rel: str, cls: str, obj: str, body: str) -> None:
    p = case / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(header(cls, obj) + body)


# ---- mesh: 3 x 3 blocks in the (y, z) cross-section, the centre one the fluid
ys = [-t, 0.0, 1.0, 1.0 + t]
counts = [m, n, m]
verts = []
index = {}
for ix, x in enumerate((0.0, L)):
    for iz, z in enumerate(ys):
        for iy, y in enumerate(ys):
            index[(ix, iy, iz)] = len(verts)
            verts.append(f"    ({x} {y} {z})")
blocks = []
faces = {"inlet": [], "outlet": [], "solidEnds": [], "outerWall": []}
for iz in range(3):
    for iy in range(3):
        v = lambda ix, dy, dz: index[(ix, iy + dy, iz + dz)]  # noqa: E731
        hexv = [v(0, 0, 0), v(1, 0, 0), v(1, 1, 0), v(0, 1, 0), v(0, 0, 1), v(1, 0, 1), v(1, 1, 1), v(0, 1, 1)]
        zone = "fluid" if (iy, iz) == (1, 1) else "solid"
        blocks.append(
            f"    hex ({' '.join(map(str, hexv))}) {zone} ({nx} {counts[iy]} {counts[iz]}) simpleGrading (1 1 1)"
        )
        ends = ("inlet", "outlet") if zone == "fluid" else ("solidEnds", "solidEnds")
        faces[ends[0]].append(f"({hexv[0]} {hexv[4]} {hexv[7]} {hexv[3]})")
        faces[ends[1]].append(f"({hexv[1]} {hexv[2]} {hexv[6]} {hexv[5]})")
        if iy == 0:
            faces["outerWall"].append(f"({hexv[0]} {hexv[1]} {hexv[5]} {hexv[4]})")
        if iy == 2:
            faces["outerWall"].append(f"({hexv[3]} {hexv[7]} {hexv[6]} {hexv[2]})")
        if iz == 0:
            faces["outerWall"].append(f"({hexv[0]} {hexv[3]} {hexv[2]} {hexv[1]})")
        if iz == 2:
            faces["outerWall"].append(f"({hexv[4]} {hexv[5]} {hexv[6]} {hexv[7]})")
types = {"inlet": "patch", "outlet": "patch", "solidEnds": "wall", "outerWall": "wall"}
boundary = "\n".join(
    f"    {name} {{ type {types[name]}; faces ({' '.join(fs)}); }}" for name, fs in faces.items()
)
write(
    "system/blockMeshDict", "dictionary", "blockMeshDict",
    "scale 1;\nvertices\n(\n" + "\n".join(verts) + "\n);\nblocks\n(\n" + "\n".join(blocks)
    + "\n);\nboundary\n(\n" + boundary + "\n);\n",
)

# ---- case level
write("system/controlDict", "dictionary", "controlDict", """regionSolvers { fluid fluid; solid solid; }
startFrom startTime; startTime 0; stopAt endTime; endTime 4000; deltaT 1;
writeControl timeStep; writeInterval 4000; purgeWrite 1;
writeFormat binary; writePrecision 12; writeCompression off;
timeFormat general; timePrecision 6; runTimeModifiable false;
""")
write("system/fvSolution", "dictionary", "fvSolution", "PIMPLE { nOuterCorrectors 1; }\n")
write("system/fvSchemes", "dictionary", "fvSchemes",
      "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\n"
      "divSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\n"
      "interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n")
write("system/decomposeParDict", "dictionary", "decomposeParDict", "numberOfSubdomains 10;\nmethod scotch;\n")

# ---- fluid region
write("system/fluid/fvSchemes", "dictionary", "fvSchemes", """ddtSchemes { default steadyState; }
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
write("system/fluid/fvSolution", "dictionary", "fvSolution", """solvers
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
write("constant/fluid/physicalProperties", "dictionary", "physicalProperties", f"""thermoType
{{
    type heRhoThermo; mixture pureMixture; transport const; thermo eConst;
    equationOfState rhoConst; specie specie; energy sensibleInternalEnergy;
}}
mixture
{{
    specie {{ molWeight 18; }}
    equationOfState {{ rho {RHO}; }}
    thermodynamics {{ Cv {CP}; hf 0; }}
    transport {{ mu {MU}; Pr {PR}; }}
}}
""")
write("constant/fluid/momentumTransport", "dictionary", "momentumTransport", "simulationType laminar;\n")
write("constant/fluid/g", "uniformDimensionedVectorField", "g", "dimensions [0 1 -2 0 0 0 0];\nvalue (0 0 0);\n")
write("constant/fluid/pRef", "uniformDimensionedScalarField", "pRef", "dimensions [1 -1 -2 0 0 0 0];\nvalue 0;\n")

fluid_iface = "fluid_to_solid"
write("0/fluid/U", "volVectorField", "U", f"""dimensions [0 1 -1 0 0 0 0];
internalField uniform (0 0 0);
boundaryField
{{
    inlet {{ type fixedValue; value uniform (1 0 0); }}
    outlet {{ type zeroGradient; }}
    {fluid_iface} {{ type noSlip; }}
}}
""")
write("0/fluid/p_rgh", "volScalarField", "p_rgh", f"""dimensions [1 -1 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type fixedFluxPressure; value uniform 0; }}
    outlet {{ type fixedValue; value uniform 0; }}
    {fluid_iface} {{ type fixedFluxPressure; value uniform 0; }}
}}
""")
write("0/fluid/p", "volScalarField", "p", f"""dimensions [1 -1 -2 0 0 0 0];
internalField uniform 0;
boundaryField
{{
    inlet {{ type calculated; value uniform 0; }}
    outlet {{ type calculated; value uniform 0; }}
    {fluid_iface} {{ type calculated; value uniform 0; }}
}}
""")
write("0/fluid/T", "volScalarField", "T", f"""dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{{
    inlet {{ type fixedValue; value uniform 300; }}
    outlet {{ type zeroGradient; }}
    {fluid_iface} {{ type coupledTemperature; value uniform 300; }}
}}
""")

# ---- solid region
write("system/solid/fvSchemes", "dictionary", "fvSchemes",
      "ddtSchemes { default steadyState; }\ngradSchemes { default Gauss linear; }\n"
      "divSchemes { default none; }\nlaplacianSchemes { default Gauss linear corrected; }\n"
      "interpolationSchemes { default linear; }\nsnGradSchemes { default corrected; }\n")
write("system/solid/fvSolution", "dictionary", "fvSolution", """solvers
{
    e { solver GAMG; smoother GaussSeidel; tolerance 1e-09; relTol 0.1; }
}
PIMPLE { nNonOrthogonalCorrectors 0; residualControl { e 1e-6; } }
relaxationFactors { equations { e 1; } }
""")
write("constant/solid/physicalProperties", "dictionary", "physicalProperties", f"""thermoType
{{
    type heSolidThermo; mixture pureMixture; transport constIsoSolid; thermo eConst;
    equationOfState rhoConst; specie specie; energy sensibleInternalEnergy;
}}
mixture
{{
    specie {{ molWeight 27; }}
    equationOfState {{ rho 1; }}
    transport {{ kappa {K_SOLID}; }}
    thermodynamics {{ hf 0; Cv 1; }}
}}
""")
write("0/solid/T", "volScalarField", "T", """dimensions [0 0 0 1 0 0 0];
internalField uniform 300;
boundaryField
{
    solidEnds { type zeroGradient; }
    outerWall { type fixedValue; value uniform 350; }
    solid_to_fluid { type coupledTemperature; value uniform 300; }
}
""")
(case / "case.info").write_text(
    f"n={n} m={m} nx={nx} fluidCells={nx*n*n} solidCells={nx*(sum(counts)**2 - n*n)} "
    f"k_fluid={K_FLUID:.6g} k_solid={K_SOLID:.6g} rhoCp={RHO*CP:.6g}\n"
)
print((case / "case.info").read_text().strip())
