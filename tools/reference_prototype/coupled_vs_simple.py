"""
coupled_vs_simple.py -- 2-D collocated finite-volume testbed: segregated
SIMPLE against a fully coupled pressure-velocity solution on a 2-D
interrupted-fin channel.  Both solvers share the SAME mesh, the SAME
discretisation (first-order upwind convection, central diffusion) and the
SAME Rhie-Chow face flux, so the outer-iteration counts isolate the
coupling strategy alone.

run_coupled() is an independent reference implementation of the block
assembly used by the C++ solver in src/coupledSimpleFoam: an implicit
pressure-gradient block in the momentum rows, and a continuity row built
from the Rhie-Chow flux that is implicit in both velocity and pressure.

References:
  Moukalled, Mangani, Darwish, "The Finite Volume Method in CFD"
    (Springer 2016) Ch. 15: collocated SIMPLE + Rhie-Chow (Secs 15.4-15.9).
  Darwish, Sraj, Moukalled, J. Comput. Phys. 228 (2009): coupled FV assembly
    (implicit pressure-gradient block in momentum; continuity row built from
    the Rhie-Chow flux, implicit in both velocity and pressure).
  ANSYS Fluent Theory Guide 23.4.3.2: pressure-based coupled algorithm;
    pseudo-transient continuation as the coupled relaxation mechanism.

Geometry: channel H=1 (48 cells), L=5 (240 cells), three interrupted fins
blocked out of the mesh.
Inlet fixed velocity; outlet fixed p=0; walls no-slip.
"""

import argparse
import json
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla

# ----------------------------------------------------------------- geometry
INTERIOR, WALL, INLET, OUTLET = 0, 1, 2, 3


class Mesh:
    def __init__(self, nx, ny, fins=True):
        self.nx, self.ny = nx, ny
        self.h = 1.0 / ny          # H = 1
        self.vol = self.h * self.h  # per-unit-depth
        self.mask = np.ones((ny, nx), dtype=bool)  # True = fluid
        if fins:
            sx = nx // 240
            sy = ny // 48
            sx = max(sx, 1)
            sy = max(sy, 1)
            for (x0, x1, y0, y1) in [(40, 90, 16, 20),
                                     (105, 155, 28, 32),
                                     (170, 220, 16, 20)]:
                self.mask[y0 * sy:y1 * sy, x0 * sx:x1 * sx] = False
        self.fluid = self.mask.ravel()
        self.n = nx * ny
        self._build_faces()

    def _build_faces(self):
        """Face arrays: owner, neigh(-1 if boundary), nx, ny components,
        type. Normal points from owner outward (to neigh / boundary)."""
        nx, ny, m = self.nx, self.ny, self.mask
        own, nb, fnx, fny, typ = [], [], [], [], []

        def add(o, n_, tx, ty, t):
            own.append(o); nb.append(n_); fnx.append(tx); fny.append(ty)
            typ.append(t)

        idx = lambda i, j: j * nx + i
        for j in range(ny):
            for i in range(nx):
                if not m[j, i]:
                    continue
                # east face
                if i == nx - 1:
                    add(idx(i, j), -1, 1, 0, OUTLET)
                elif m[j, i + 1]:
                    add(idx(i, j), idx(i + 1, j), 1, 0, INTERIOR)
                else:
                    add(idx(i, j), -1, 1, 0, WALL)
                # west face only when boundary/solid (interior counted once)
                if i == 0:
                    add(idx(i, j), -1, -1, 0, INLET)
                elif not m[j, i - 1]:
                    add(idx(i, j), -1, -1, 0, WALL)
                # north face
                if j == ny - 1:
                    add(idx(i, j), -1, 0, 1, WALL)
                elif m[j + 1, i]:
                    add(idx(i, j), idx(i, j + 1), 0, 1, INTERIOR)
                else:
                    add(idx(i, j), -1, 0, 1, WALL)
                # south face only when boundary/solid
                if j == 0:
                    add(idx(i, j), -1, 0, -1, WALL)
                elif not m[j - 1, i]:
                    add(idx(i, j), -1, 0, -1, WALL)

        self.f_own = np.array(own)
        self.f_nb = np.array(nb)
        self.f_nx = np.array(fnx, dtype=float)
        self.f_ny = np.array(fny, dtype=float)
        self.f_typ = np.array(typ)
        self.area = self.h  # uniform, per unit depth
        self.int_f = self.f_typ == INTERIOR
        self.wall_f = self.f_typ == WALL
        self.in_f = self.f_typ == INLET
        self.out_f = self.f_typ == OUTLET


# --------------------------------------------------------------- operators
class Discretization:
    """Shared pieces: gradients, momentum coefficients, Rhie-Chow flux."""

    def __init__(self, mesh, rho, mu, u_in):
        self.msh = mesh
        self.rho, self.mu, self.u_in = rho, mu, u_in

    def grad_p(self, p):
        """Green-Gauss cell gradient. Face p: interior mean; wall/inlet
        zero-grad (p_own); outlet Dirichlet 0."""
        msh = self.msh
        A = msh.area
        pf = np.zeros(msh.f_own.size)
        io = msh.int_f
        pf[io] = 0.5 * (p[msh.f_own[io]] + p[msh.f_nb[io]])
        zg = msh.wall_f | msh.in_f
        pf[zg] = p[msh.f_own[zg]]
        pf[msh.out_f] = 0.0
        gx = np.zeros(msh.n)
        gy = np.zeros(msh.n)
        np.add.at(gx, msh.f_own, pf * msh.f_nx * A)
        np.add.at(gy, msh.f_own, pf * msh.f_ny * A)
        io_nb = msh.f_nb[io]
        np.add.at(gx, io_nb, -pf[io] * msh.f_nx[io] * A)
        np.add.at(gy, io_nb, -pf[io] * msh.f_ny[io] * A)
        return gx / msh.vol, gy / msh.vol

    def momentum_coeffs(self, mf, dtau_diag=None, relax=None, u_old=None,
                        v_old=None):
        """Assemble A (same for u and v), plus BC RHS pieces.
        Returns (A_sparse, bu_bc, bv_bc, aP_used).
        Upwind convection with face mass flux mf (positive = out of owner);
        central diffusion; Dirichlet half-cell diffusion on walls/inlet.
        relax=(alpha, which applies Patankar implicit relaxation and returns
        the extra RHS terms separately via bu/bv closures on u_old/v_old."""
        msh, mu, rho = self.msh, self.mu, self.rho
        A, h = msh.area, msh.h
        n = msh.n
        rows, cols, vals = [], [], []
        aP = np.zeros(n)
        bu = np.zeros(n)
        bv = np.zeros(n)

        # interior faces: owner and neighbour sides
        io = msh.int_f
        o, nb2 = msh.f_own[io], msh.f_nb[io]
        m = mf[io]
        dcoef = mu * A / h
        # owner row: aP += D + max(m,0); aN = -(D + max(-m,0))
        a_on = dcoef + np.maximum(-m, 0.0)
        a_no = dcoef + np.maximum(m, 0.0)
        np.add.at(aP, o, dcoef + np.maximum(m, 0.0))
        np.add.at(aP, nb2, dcoef + np.maximum(-m, 0.0))
        rows += [o, nb2]
        cols += [nb2, o]
        vals += [-a_on, -a_no]

        # wall faces: no-slip both components (Dirichlet 0, half cell)
        w = msh.wall_f
        ow = msh.f_own[w]
        np.add.at(aP, ow, mu * A / (h / 2.0))

        # inlet faces: Dirichlet (u_in, 0); mf is negative (inflow)
        ii = msh.in_f
        oi = msh.f_own[ii]
        m_in = mf[ii]
        dh = mu * A / (h / 2.0)
        np.add.at(aP, oi, dh + np.maximum(m_in, 0.0))
        np.add.at(bu, oi, (dh + np.maximum(-m_in, 0.0)) * self.u_in)
        # v inlet value is 0 -> no RHS

        # outlet: zero-grad diffusion, upwind convection (outflow uses uC)
        oo = msh.f_own[msh.out_f]
        m_out = mf[msh.out_f]
        np.add.at(aP, oo, np.maximum(m_out, 0.0))
        # (inflow at outlet would need a value; assume none at converged state)

        aP_used = aP.copy()
        if dtau_diag is not None:
            aP_used = aP_used + dtau_diag
            bu += dtau_diag * (u_old if u_old is not None else 0.0)
            bv += dtau_diag * (v_old if v_old is not None else 0.0)
        if relax is not None:
            alpha = relax
            bu += (1.0 - alpha) / alpha * aP_used * u_old
            bv += (1.0 - alpha) / alpha * aP_used * v_old
            aP_used = aP_used / alpha

        # solid cells: identity rows
        solid = ~msh.fluid
        aP_used[solid] = 1.0
        rows.append(np.arange(n))
        cols.append(np.arange(n))
        vals.append(aP_used)
        Amat = sp.csr_matrix(
            (np.concatenate(vals),
             (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        # zero out solid off-diagonals happens naturally (solid cells have no
        # faces); their rhs:
        bu[solid] = 0.0
        bv[solid] = 0.0
        return Amat, bu, bv, aP_used

    def face_D(self, aP):
        """Rhie-Chow D_f = interpolated (V/aP)."""
        msh = self.msh
        Dc = np.where(msh.fluid, msh.vol / aP, 0.0)
        Df = np.zeros(msh.f_own.size)
        io = msh.int_f
        Df[io] = 0.5 * (Dc[msh.f_own[io]] + Dc[msh.f_nb[io]])
        Df[msh.out_f] = Dc[msh.f_own[msh.out_f]]
        return Df

    def face_flux(self, u, v, p, Df, gpx, gpy):
        """Rhie-Chow mass flux on every face (positive out of owner)."""
        msh, rho = self.msh, self.rho
        A, h = msh.area, msh.h
        mf = np.zeros(msh.f_own.size)
        io = msh.int_f
        o, nb2 = msh.f_own[io], msh.f_nb[io]
        un = 0.5 * ((u[o] + u[nb2]) * msh.f_nx[io] +
                    (v[o] + v[nb2]) * msh.f_ny[io])
        dpdn = (p[nb2] - p[o]) / h
        gpn = 0.5 * ((gpx[o] + gpx[nb2]) * msh.f_nx[io] +
                     (gpy[o] + gpy[nb2]) * msh.f_ny[io])
        mf[io] = rho * A * (un - Df[io] * (dpdn - gpn))
        # inlet: fixed
        mf[msh.in_f] = -rho * A * self.u_in
        # outlet: zero-grad velocity + RC with Dirichlet p (half cell)
        oo = msh.f_own[msh.out_f]
        dpdn_o = (0.0 - p[oo]) / (h / 2.0)
        mf[msh.out_f] = rho * A * (u[oo] - Df[msh.out_f] *
                                   (dpdn_o - gpx[oo]))
        # walls stay 0
        return mf

    def mass_imbalance(self, mf):
        msh = self.msh
        div = np.zeros(msh.n)
        np.add.at(div, msh.f_own, mf)
        io = msh.int_f
        np.add.at(div, msh.f_nb[io], -mf[io])
        div[~msh.fluid] = 0.0
        return div

    def momentum_residual(self, u, v, p, mf):
        """Unrelaxed momentum residual (same definition for both solvers)."""
        Amat, bu, bv, aP = self.momentum_coeffs(mf)
        gpx, gpy = self.grad_p(p)
        ru = bu - self.msh.vol * gpx - Amat @ u
        rv = bv - self.msh.vol * gpy - Amat @ v
        f = self.msh.fluid
        norm = self.rho * self.u_in ** 2 * self.msh.area * self.msh.ny
        return (np.abs(ru[f]).sum() + np.abs(rv[f]).sum()) / norm


# ------------------------------------------------------------------ SIMPLE
def run_simple(disc, alpha_u=0.7, alpha_p=0.3, tol_mom=1e-5, tol_mass=1e-6,
               max_it=6000, log_every=200):
    msh = disc.msh
    rho, h, A = disc.rho, msh.h, msh.area
    n = msh.n
    u = np.where(msh.fluid, disc.u_in, 0.0)
    v = np.zeros(n)
    p = np.zeros(n)
    gpx, gpy = disc.grad_p(p)
    # initial flux from interpolation only
    _, _, _, aP0 = disc.momentum_coeffs(np.zeros(msh.f_own.size))
    Df = disc.face_D(aP0)
    mf = disc.face_flux(u, v, p, Df, gpx, gpy)
    mdot_in = rho * A * disc.u_in * np.count_nonzero(
        msh.mask[:, 0])
    hist = []
    t0 = time.perf_counter()
    r_mom0 = None
    for it in range(1, max_it + 1):
        Amat, bu, bv, aP = disc.momentum_coeffs(
            mf, relax=alpha_u, u_old=u, v_old=v)
        gpx, gpy = disc.grad_p(p)
        u = spla.spsolve(Amat, bu - msh.vol * gpx)
        v = spla.spsolve(Amat, bv - msh.vol * gpy)
        Df = disc.face_D(aP)
        mf = disc.face_flux(u, v, p, Df, gpx, gpy)
        div = disc.mass_imbalance(mf)
        # pressure correction equation
        rows, cols, vals = [], [], []
        diag = np.zeros(n)
        io = msh.int_f
        o, nb2 = msh.f_own[io], msh.f_nb[io]
        coef = rho * A * Df[io] / h
        np.add.at(diag, o, coef)
        np.add.at(diag, nb2, coef)
        rows += [o, nb2]; cols += [nb2, o]; vals += [-coef, -coef]
        oo = msh.f_own[msh.out_f]
        np.add.at(diag, oo, rho * A * Df[msh.out_f] / (h / 2.0))
        diag[~msh.fluid] = 1.0
        diag[diag == 0.0] = 1.0
        rows.append(np.arange(n)); cols.append(np.arange(n)); vals.append(diag)
        Ap = sp.csr_matrix((np.concatenate(vals),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n, n))
        rhs = -div
        rhs[~msh.fluid] = 0.0
        pc = spla.spsolve(Ap, rhs)
        # corrections
        gcx, gcy = disc.grad_p_correction(pc) if hasattr(
            disc, 'grad_p_correction') else disc.grad_p(pc)
        Dc = np.where(msh.fluid, msh.vol / aP, 0.0)
        u = u - Dc * gcx
        v = v - Dc * gcy
        p = p + alpha_p * pc
        gpx, gpy = disc.grad_p(p)
        mf_corr = np.zeros_like(mf)
        mf_corr[io] = -rho * A * Df[io] * (pc[nb2] - pc[o]) / h
        mf_corr[msh.out_f] = -rho * A * Df[msh.out_f] * \
            (0.0 - pc[oo]) / (h / 2.0)
        mf = mf + mf_corr
        # residuals
        r_mass = np.abs(disc.mass_imbalance(mf)).sum() / mdot_in
        r_mom = disc.momentum_residual(u, v, p, mf)
        if r_mom0 is None:
            r_mom0 = max(r_mom, 1e-30)
        hist.append((it, r_mom / r_mom0, r_mass))
        if it % log_every == 0:
            print(f"    SIMPLE it {it}: mom {r_mom/r_mom0:.3e} "
                  f"mass {r_mass:.3e}")
        if r_mom / r_mom0 < tol_mom and r_mass < tol_mass:
            break
        if not np.isfinite(r_mom) or r_mom / r_mom0 > 1e6:
            print("    SIMPLE diverged")
            return dict(conv=False, it=it, t=time.perf_counter() - t0,
                        hist=hist)
    t1 = time.perf_counter()
    return dict(conv=(r_mom / r_mom0 < tol_mom and r_mass < tol_mass),
                it=it, t=t1 - t0, u=u, v=v, p=p, mf=mf, hist=hist,
                dp=inlet_dp(disc, p))


# ----------------------------------------------------------------- coupled
def run_coupled(disc, tol_mom=1e-5, tol_mass=1e-6, max_it=200,
                pseudo_cfl=None, log_every=5):
    msh = disc.msh
    rho, h, A = disc.rho, msh.h, msh.area
    n = msh.n
    u = np.where(msh.fluid, disc.u_in, 0.0)
    v = np.zeros(n)
    p = np.zeros(n)
    gpx, gpy = disc.grad_p(p)
    _, _, _, aP0 = disc.momentum_coeffs(np.zeros(msh.f_own.size))
    Df = disc.face_D(aP0)
    mf = disc.face_flux(u, v, p, Df, gpx, gpy)
    mdot_in = rho * A * disc.u_in * np.count_nonzero(msh.mask[:, 0])
    hist = []
    r_mom0 = None
    t0 = time.perf_counter()
    for it in range(1, max_it + 1):
        dtau_diag = None
        if pseudo_cfl is not None:
            vel = np.maximum(np.sqrt(u * u + v * v), 0.05 * disc.u_in)
            dtau = pseudo_cfl * h / vel
            dtau_diag = rho * msh.vol / dtau
        Amat, bu, bv, aP = disc.momentum_coeffs(
            mf, dtau_diag=dtau_diag, u_old=u, v_old=v)
        gpx, gpy = disc.grad_p(p)
        Df = disc.face_D(aP)

        # ---- pressure-gradient blocks Gx, Gy (Green-Gauss, implicit)
        rows, cols, vals_x, vals_y = [], [], [], []
        io = msh.int_f
        o, nb2 = msh.f_own[io], msh.f_nb[io]
        rows += [o, o, nb2, nb2]
        cols += [o, nb2, nb2, o]
        for vv, fn in ((vals_x, msh.f_nx), (vals_y, msh.f_ny)):
            c_of = 0.5 * fn[io] * A
            vv += [c_of, c_of, -c_of, -c_of]
        zg = msh.wall_f | msh.in_f
        oz = msh.f_own[zg]
        rows.append(oz); cols.append(oz)
        vals_x.append(msh.f_nx[zg] * A)
        vals_y.append(msh.f_ny[zg] * A)
        # outlet p=0: no implicit contribution
        Gx = sp.csr_matrix((np.concatenate(vals_x),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n, n))
        Gy = sp.csr_matrix((np.concatenate(vals_y),
                            (np.concatenate(rows), np.concatenate(cols))),
                           shape=(n, n))
        # solid rows of G must be empty
        fluid_diag = sp.diags(msh.fluid.astype(float))
        Gx = fluid_diag @ Gx
        Gy = fluid_diag @ Gy

        # ---- continuity row: Du u + Dv v + App p = b_p
        rows_c, cols_c, vals_cu, vals_cv = [], [], [], []
        rows_p, cols_p, vals_p = [], [], []
        b_p = np.zeros(n)
        # interior faces
        cu = 0.5 * rho * A * msh.f_nx[io]
        cv = 0.5 * rho * A * msh.f_ny[io]
        # owner row gets +, neighbour row gets - (flux out of owner)
        rows_c += [o, o, nb2, nb2]
        cols_c += [o, nb2, o, nb2]
        vals_cu += [cu, cu, -cu, -cu]
        vals_cv += [cv, cv, -cv, -cv]
        cp = rho * A * Df[io] / h
        rows_p += [o, o, nb2, nb2]
        cols_p += [o, nb2, nb2, o]
        vals_p += [cp, -cp, cp, -cp]
        gpn = 0.5 * ((gpx[o] + gpx[nb2]) * msh.f_nx[io] +
                     (gpy[o] + gpy[nb2]) * msh.f_ny[io])
        expl = rho * A * Df[io] * gpn
        np.add.at(b_p, o, -expl)
        np.add.at(b_p, nb2, expl)
        # inlet: fixed inflow to RHS
        oi = msh.f_own[msh.in_f]
        np.add.at(b_p, oi, rho * A * disc.u_in)
        # outlet: rho*A*u_C  - rho*A*Df*((0-p_C)/(h/2) - gpx_C)
        oo = msh.f_own[msh.out_f]
        rows_c.append(oo); cols_c.append(oo)
        vals_cu.append(rho * A * np.ones(oo.size))
        vals_cv.append(np.zeros(oo.size))
        cpo = rho * A * Df[msh.out_f] / (h / 2.0)
        rows_p.append(oo); cols_p.append(oo); vals_p.append(cpo)
        np.add.at(b_p, oo, -rho * A * Df[msh.out_f] * gpx[oo])
        Du = sp.csr_matrix((np.concatenate(vals_cu),
                            (np.concatenate(rows_c), np.concatenate(cols_c))),
                           shape=(n, n))
        Dv = sp.csr_matrix((np.concatenate(vals_cv),
                            (np.concatenate(rows_c), np.concatenate(cols_c))),
                           shape=(n, n))
        App = sp.csr_matrix((np.concatenate(vals_p),
                             (np.concatenate(rows_p), np.concatenate(cols_p))),
                            shape=(n, n))
        # solid cells: p identity row, zero elsewhere
        solid = ~msh.fluid
        Du = fluid_diag @ Du @ fluid_diag
        Dv = fluid_diag @ Dv @ fluid_diag
        App = fluid_diag @ App + sp.diags(solid.astype(float))
        b_p[solid] = 0.0

        big = sp.bmat([[Amat, None, Gx],
                       [None, Amat, Gy],
                       [Du, Dv, App]], format='csc')
        rhs = np.concatenate([bu, bv, b_p])
        sol = spla.splu(big).solve(rhs)
        u, v, p = sol[:n], sol[n:2 * n], sol[2 * n:]
        u[solid] = 0.0
        v[solid] = 0.0

        gpx, gpy = disc.grad_p(p)
        mf = disc.face_flux(u, v, p, Df, gpx, gpy)
        r_mass = np.abs(disc.mass_imbalance(mf)).sum() / mdot_in
        r_mom = disc.momentum_residual(u, v, p, mf)
        if r_mom0 is None:
            r_mom0 = max(r_mom, 1e-30)
        hist.append((it, r_mom / r_mom0, r_mass))
        if it % log_every == 0 or it <= 3:
            print(f"    COUPLED it {it}: mom {r_mom/r_mom0:.3e} "
                  f"mass {r_mass:.3e}")
        if r_mom / r_mom0 < tol_mom and r_mass < tol_mass:
            break
        if not np.isfinite(r_mom) or r_mom / r_mom0 > 1e8:
            print("    COUPLED diverged")
            return dict(conv=False, it=it, t=time.perf_counter() - t0,
                        hist=hist)
    t1 = time.perf_counter()
    return dict(conv=(r_mom / r_mom0 < tol_mom and r_mass < tol_mass),
                it=it, t=t1 - t0, u=u, v=v, p=p, mf=mf, hist=hist,
                dp=inlet_dp(disc, p))


def inlet_dp(disc, p):
    msh = disc.msh
    oi = msh.f_own[msh.in_f]
    return float(np.mean(p[oi]))  # outlet is 0 by construction


# --------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--nx', type=int, default=240)
    ap.add_argument('--ny', type=int, default=48)
    ap.add_argument('--re', type=float, default=400.0)
    ap.add_argument('--quick', action='store_true')
    args = ap.parse_args()
    if args.quick:
        args.nx, args.ny = 120, 24

    rho, u_in, H = 1.0, 1.0, 1.0
    mu = rho * u_in * H / args.re
    msh = Mesh(args.nx, args.ny)
    disc = Discretization(msh, rho, mu, u_in)
    print(f"mesh {args.nx}x{args.ny}, fluid cells "
          f"{int(msh.fluid.sum())}, Re={args.re:g}")

    results = {}
    for name, fn in [
        ('SIMPLE a_u=0.7 a_p=0.3',
         lambda: run_simple(disc, 0.7, 0.3)),
        ('SIMPLE a_u=0.85 a_p=0.5',
         lambda: run_simple(disc, 0.85, 0.5)),
        ('COUPLED pure',
         lambda: run_coupled(disc)),
        ('COUPLED pseudo-transient CFL=200',
         lambda: run_coupled(disc, pseudo_cfl=200.0)),
    ]:
        print(f"\n=== {name} ===")
        r = fn()
        results[name] = dict(conv=bool(r['conv']), it=int(r['it']),
                             t=float(r['t']))
        if 'dp' in r:
            results[name]['dp'] = float(r['dp'])
        if r.get('hist'):
            results[name]['final_mom'] = float(r['hist'][-1][1])
            results[name]['final_mass'] = float(r['hist'][-1][2])
        print(f"  -> conv={r['conv']} iters={r['it']} "
              f"wall={r['t']:.2f}s dp={r.get('dp', float('nan')):.6f}")

    out = f"results_re{int(args.re)}_{args.nx}x{args.ny}.json"
    with open(out, 'w') as fh:
        json.dump(results, fh, indent=2)
    print(f"\nwrote {out}")


if __name__ == '__main__':
    main()
