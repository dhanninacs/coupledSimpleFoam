#!/usr/bin/env bash
# Make and run the conjugate duct with foamMultiRun (the CPU reference): cht_run.sh <name> <n> <np> [m]
export USER="${USER:-$(id -un)}"
source /opt/openfoam14/etc/bashrc >/dev/null 2>&1 || true
W="$(cd "$(dirname "$0")" && pwd)"
name=$1 n=$2 np=${3:-1} m=${4:-}
d="$HOME/gpu-bench/$name"
rm -rf "$d"
python3 "$W/cht_case.py" "$d" "$n" $m || exit 1
cd "$d" || exit 1
blockMesh > log.blockMesh 2>&1 || { echo "blockMesh failed"; tail -20 log.blockMesh; exit 1; }
splitMeshRegions -cellZones "(fluid solid)" -noFields > log.split 2>&1 || { echo "split failed"; tail -20 log.split; exit 1; }
ls constant/fluid/polyMesh constant/solid/polyMesh > /dev/null || { echo "regions missing"; exit 1; }
grep -E "fluid_to_solid|solid_to_fluid" constant/fluid/polyMesh/boundary constant/solid/polyMesh/boundary | head -2
sed -i "s/numberOfSubdomains [0-9]*;/numberOfSubdomains $np;/" system/decomposeParDict
start=$(date +%s.%N)
if [ "$np" -gt 1 ]; then
  decomposePar -allRegions -force > log.decomposePar 2>&1 || { echo "decomposePar failed"; tail -20 log.decomposePar; exit 1; }
  start=$(date +%s.%N)
  mpirun --bind-to core -np "$np" foamMultiRun -parallel > log.run 2>&1; rc=$?
  end=$(date +%s.%N)
  reconstructPar -allRegions -latestTime > log.reconstruct 2>&1
else
  foamMultiRun > log.run 2>&1; rc=$?
  end=$(date +%s.%N)
fi
wall=$(python3 -c "print(f'{$end-$start:.1f}')")
echo "cht case=$name np=$np rc=$rc wall=${wall}s iterations=$(grep -c '^Time = ' log.run) $(grep -oE 'converged in [0-9]+ iterations' log.run | tail -1)"
grep -E 'FOAM FATAL|nan' log.run | head -3
