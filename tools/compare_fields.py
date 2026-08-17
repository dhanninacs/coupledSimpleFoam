#!/usr/bin/env python3
"""Compare two OpenFOAM solutions cell by cell on a structured 2-D mesh.

Reads the ASCII internalField of U from two case/time directories and
reports the largest cell-wise velocity difference anywhere in the domain,
then the profile of Ux along the vertical centreline.

The mesh is assumed to be a single structured blockMesh hex block
(n x n x 1), as in the lid-driven cavity tutorials, so cell index = j*n + i
with i the fastest index.  The vertical centreline of an even-n mesh falls
on a face, so both adjacent columns (i = n/2-1, n/2) are reported.

Typical uses: coupled against segregated, parallel against serial, or one
linear-algebra backend against another.

Usage:
    compare_fields.py <caseA> <timeA> <caseB> <timeB> [n]

n is the mesh size and defaults to 64.
"""

import re
import sys


def read_field(path):
    """Return the internalField of an OpenFOAM ascii vol field as a list of
    tuples (scalars come back as 1-tuples)."""
    with open(path) as fh:
        txt = fh.read()

    m = re.search(r"internalField\s+nonuniform\s+List<(\w+)>\s*\n?\s*(\d+)\s*\(",
                  txt)
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
        raise RuntimeError("%s holds a uniform field" % path)
    m = re.search(r"internalField\s+uniform\s+([-\deE.+]+)", txt)
    if m:
        raise RuntimeError("%s holds a uniform field" % path)
    raise RuntimeError("cannot parse %s" % path)


def main():
    if len(sys.argv) < 5:
        print(__doc__)
        return 1

    caseA, timeA, caseB, timeB = sys.argv[1:5]
    n = int(sys.argv[5]) if len(sys.argv) > 5 else 64

    UA = read_field("%s/%s/U" % (caseA, timeA))
    UB = read_field("%s/%s/U" % (caseB, timeB))

    if len(UA) != len(UB):
        raise RuntimeError("cell counts differ: %d vs %d" % (len(UA), len(UB)))

    gmaxUx, cmaxUx = max((abs(a[0] - b[0]), k)
                         for k, (a, b) in enumerate(zip(UA, UB)))
    gmaxUy = max(abs(a[1] - b[1]) for a, b in zip(UA, UB))
    gmaxU = max(
        ((a[0] - b[0])**2 + (a[1] - b[1])**2 + (a[2] - b[2])**2)**0.5
        for a, b in zip(UA, UB)
    )

    print("cells                       : %d" % len(UA))
    print("max |Ux_A - Ux_B|  (all)    : %.6e  at cell %d "
          "(x=%.5f, y=%.5f)"
          % (gmaxUx, cmaxUx, (cmaxUx % n + 0.5)/n, (cmaxUx//n + 0.5)/n))
    print("max |Uy_A - Uy_B|  (all)    : %.6e" % gmaxUy)
    print("max |U_A  - U_B |  (all)    : %.6e" % gmaxU)

    for icol in (n // 2 - 1, n // 2):
        xc = (icol + 0.5)/n
        dmax = 0.0
        jmax = -1
        for j in range(n):
            c = j*n + icol
            d = abs(UA[c][0] - UB[c][0])
            if d > dmax:
                dmax, jmax = d, j
        print("centreline x=%.7f  max |dUx| = %.6e   at y=%.7f"
              % (xc, dmax, (jmax + 0.5)/n))

    print()
    print("Ux(y) on the vertical centreline (mean of the two columns)")
    print("%12s %18s %18s %14s" % ("y", "Ux_A", "Ux_B", "diff"))
    for j in range(0, n, max(1, n // 16)):
        c1 = j*n + n//2 - 1
        c2 = j*n + n//2
        a = 0.5*(UA[c1][0] + UA[c2][0])
        b = 0.5*(UB[c1][0] + UB[c2][0])
        print("%12.7f %18.10e %18.10e %14.3e"
              % ((j + 0.5)/n, a, b, a - b))

    return 0


if __name__ == '__main__':
    sys.exit(main())
