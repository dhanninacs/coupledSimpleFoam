#!/usr/bin/env python3
"""Local Nusselt number of a plane channel from an OpenFOAM solution.

Reads the ASCII internalFields of T, U and the cell-centre field C from one
case/time directory and reports, at each requested streamwise
station, the bulk temperature, the wall temperature gradient and

    Nu = |dT/dy|_wall * Dh / (Tw - Tb),      Dh = 2*H

which is the local Nusselt number of an infinite parallel-plate duct.  Both
the thermal conductivity and rho*cp cancel out of that expression, so nothing
beyond the temperature and velocity fields is needed - which is what makes it
a test of the discrete temperature solution alone.

For fully developed laminar flow between plates held at a constant and equal
temperature the exact value is Nu = 7.5407 (Shah & London).  The comparison is
only meaningful where the flow and the temperature profile are both developed
and where axial conduction is negligible, i.e. several hydraulic diameters
downstream and at Pe = Re*Pr of order 100 or more.

The cell centres come from the volVectorField C written by

    foamPostProcess -latestTime -func writeCellCentres

so no mesh dimension has to be repeated on the command line.  (That function
object also writes the components as Ccx/Ccy/Ccz; C is read here instead
because its name has been stable across OpenFOAM versions.)  The mesh must be
a structured tensor-product grid, uniform in y, one cell thick - which is what
tutorials/heatedDuct provides.

Usage:
    heated_duct_nusselt.py <case> <time> <Twall> [x1 x2 ...]

With no stations given, four are chosen automatically in the downstream half
of the domain.
"""

import re
import sys


NU_EXACT = 7.5407


def read_field(path):
    """Return the internalField of an OpenFOAM ascii volScalarField or
    volVectorField as a list of tuples (scalars come back as 1-tuples)."""
    with open(path) as fh:
        txt = fh.read()

    m = re.search(
        r"internalField\s+nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\(", txt)
    if m:
        n = int(m.group(2))
        start = m.end()
        depth = 1
        i = start
        while depth:
            if txt[i] == '(':
                depth += 1
            elif txt[i] == ')':
                depth -= 1
            i += 1
        body = txt[start:i - 1]

        if m.group(1) == 'scalar':
            vals = [(float(v),) for v in body.split()]
        else:
            vals = [tuple(float(x) for x in t.split())
                    for t in re.findall(r"\(([^()]*)\)", body)]

        if len(vals) != n:
            raise RuntimeError("%s: expected %d entries, parsed %d"
                               % (path, n, len(vals)))
        return vals

    m = re.search(r"internalField\s+uniform\s+\(([^)]*)\)", txt)
    if m:
        v = tuple(float(x) for x in m.group(1).split())
        raise RuntimeError("%s holds the uniform field %s; the case has not "
                           "been solved" % (path, v))
    m = re.search(r"internalField\s+uniform\s+([-\deE.+]+)", txt)
    if m:
        raise RuntimeError("%s holds a uniform field; the case has not been "
                           "solved" % path)
    raise RuntimeError("cannot parse %s" % path)


def columns(C):
    """Group cell indices into streamwise stations.

    Returns (xs, cols) with xs the sorted distinct x of the cell centres and
    cols[k] the cell indices of station k, ordered by increasing y."""
    # Round before grouping: cell centres in one column are written to the
    # same digits but need not be bit-identical after arithmetic, and a
    # split column would silently halve the bulk-temperature integral.
    order = {}
    for c, p in enumerate(C):
        order.setdefault(round(p[0], 9), []).append(c)

    xs = sorted(order)
    cols = [sorted(order[x], key=lambda c: C[c][1]) for x in xs]

    n = len(cols[0])
    for col in cols:
        if len(col) != n:
            raise RuntimeError("the mesh is not a tensor-product grid: "
                               "stations hold %d and %d cells"
                               % (n, len(col)))
    return xs, cols


def main():
    if len(sys.argv) < 4:
        print(__doc__)
        return 1

    case, time = sys.argv[1], sys.argv[2]
    Tw = float(sys.argv[3])
    wanted = [float(a) for a in sys.argv[4:]]

    T = read_field("%s/%s/T" % (case, time))
    U = read_field("%s/%s/U" % (case, time))
    C = read_field("%s/%s/C" % (case, time))

    if not (len(T) == len(U) == len(C)):
        raise RuntimeError("field sizes differ: T %d, U %d, C %d"
                           % (len(T), len(U), len(C)))

    xs, cols = columns(C)

    ys = [C[c][1] for c in cols[0]]
    dy = ys[1] - ys[0]
    spread = max(abs((ys[j + 1] - ys[j]) - dy) for j in range(len(ys) - 1))
    if spread > 1e-9*dy:
        raise RuntimeError("the mesh is graded in y (spacing varies by %.3e); "
                           "this script assumes a uniform y spacing" % spread)

    # The walls sit half a cell outside the first and last cell centres
    yLo = ys[0] - 0.5*dy
    yHi = ys[-1] + 0.5*dy
    H = yHi - yLo
    Dh = 2.0*H

    if not wanted:
        L = xs[-1] + 0.5*(xs[1] - xs[0])
        wanted = [0.55*L, 0.70*L, 0.85*L, 0.99*L]

    print("cells                 : %d  (%d x %d)"
          % (len(T), len(xs), len(ys)))
    print("channel height H      : %.6f m   (y from %.6f to %.6f)"
          % (H, yLo, yHi))
    print("hydraulic diameter Dh : %.6f m" % Dh)
    print("wall temperature      : %.6f" % Tw)
    print()
    print("%10s %16s %16s %16s %10s %9s"
          % ("x", "Tb", "Tw - Tb", "|dT/dy|_wall", "Nu", "Nu/7.5407"))

    out = []
    for xr in wanted:
        k = min(range(len(xs)), key=lambda i: abs(xs[i] - xr))
        col = cols[k]

        num = sum(U[c][0]*T[c][0] for c in col)
        den = sum(U[c][0] for c in col)
        Tb = num/den

        # One-sided wall gradient.  Second-order accurate here because the
        # temperature profile has zero curvature at a no-slip wall.
        gLo = (Tw - T[col[0]][0])/(0.5*dy)
        gHi = (Tw - T[col[-1]][0])/(0.5*dy)
        g = 0.5*(abs(gLo) + abs(gHi))

        Nu = g*Dh/(Tw - Tb)
        out.append((xs[k], Nu))

        print("%10.4f %16.8f %16.8f %16.6f %10.4f %9.5f"
              % (xs[k], Tb, Tw - Tb, g, Nu, Nu/NU_EXACT))

    print()
    xLast, NuLast = out[-1]
    print("Nu at x = %.4f : %.4f   (exact fully developed value %.4f, "
          "%+.2f %%)"
          % (xLast, NuLast, NU_EXACT, 100.0*(NuLast - NU_EXACT)/NU_EXACT))

    return 0


if __name__ == '__main__':
    sys.exit(main())
