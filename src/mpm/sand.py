"""MLS-MPM sand solver with simplified Drucker-Prager plasticity.

P2G → grid update → G2P → particle advection. The plasticity step is a
return-mapping on the singular values of the elastic part of F (Stomakhin 2013-style),
projected onto the Drucker-Prager yield surface.

This is a *baseline* implementation — sufficient for Step 2 verification (sand
piles up in a box at game framerate). Replaceable by PB-MPM (Step 11).
"""

from __future__ import annotations

import taichi as ti

from .grid import Grid


@ti.data_oriented
class SandSolver:
    """MLS-MPM sand with Drucker-Prager plasticity."""

    def __init__(
        self,
        particles: ti.StructField,
        grid: Grid,
        E: float = 1.0e5,         # Young's modulus
        nu: float = 0.3,          # Poisson ratio
        friction_angle_deg: float = 30.0,
    ):
        self.particles = particles
        self.grid = grid
        self.dx = grid.dx
        self.inv_dx = grid.inv_dx

        # Lamé parameters (kept hot in fields so kernels can read)
        mu = E / (2.0 * (1.0 + nu))
        la = E * nu / ((1.0 + nu) * (1.0 - 2.0 * nu))
        self.mu = ti.field(ti.f32, shape=())
        self.la = ti.field(ti.f32, shape=())
        self.mu[None] = mu
        self.la[None] = la

        # Drucker-Prager friction coefficient: α = (2/sqrt(3)) sin φ / (3 - sin φ)
        import math
        sin_phi = math.sin(math.radians(friction_angle_deg))
        self.alpha = ti.field(ti.f32, shape=())
        self.alpha[None] = (2.0 / math.sqrt(3.0)) * sin_phi / (3.0 - sin_phi)

        # Default per-particle volume (used in stress computation)
        self.p_vol = ti.field(ti.f32, shape=())
        self.p_vol[None] = (self.dx * 0.5) ** 3

    @ti.kernel
    def p2g(self):
        """Particle → grid: scatter mass and (mass*velocity + APIC affine) into nodes."""
        for p in range(self.particles.shape[0]):
            if self.particles[p].active == 0:
                continue
            xp = self.particles[p].pos * self.inv_dx
            base = ti.cast(xp - 0.5, ti.i32)
            fx = xp - ti.cast(base, ti.f32)
            # Quadratic B-spline weights
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            # Constitutive — neo-Hookean elastic stress + Drucker-Prager plasticity
            F = self.particles[p].F
            U, sig, V = ti.svd(F, ti.f32)
            J = sig[0, 0] * sig[1, 1] * sig[2, 2]
            mu = self.mu[None]
            la = self.la[None]
            # Cauchy stress: σ = (1/J)·(2μ(F-RU)F^T + λ·log(J)·I)·F^T  ; here a simpler form:
            R = U @ V.transpose()
            stress = 2.0 * mu * (F - R) @ F.transpose() + ti.Matrix.identity(ti.f32, 3) * la * (J - 1.0) * J
            stress *= -self.p_vol[None] * 4.0 * self.inv_dx * self.inv_dx
            affine = stress + self.particles[p].mass * self.particles[p].C

            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                idx = base + offset
                if 0 <= idx[0] < self.grid.res and 0 <= idx[1] < self.grid.res and 0 <= idx[2] < self.grid.res:
                    self.grid.node_mass[idx] += weight * self.particles[p].mass
                    self.grid.node_mom[idx] += weight * (
                        self.particles[p].mass * self.particles[p].vel + affine @ dpos
                    )

    @ti.kernel
    def g2p(self, dt: ti.f32):
        """Grid → particle: gather velocity + APIC affine, advect."""
        for p in range(self.particles.shape[0]):
            if self.particles[p].active == 0:
                continue
            xp = self.particles[p].pos * self.inv_dx
            base = ti.cast(xp - 0.5, ti.i32)
            fx = xp - ti.cast(base, ti.f32)
            w = [
                0.5 * (1.5 - fx) ** 2,
                0.75 - (fx - 1.0) ** 2,
                0.5 * (fx - 0.5) ** 2,
            ]

            new_v = ti.Vector([0.0, 0.0, 0.0])
            new_C = ti.Matrix.zero(ti.f32, 3, 3)
            for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
                offset = ti.Vector([i, j, k])
                dpos = (ti.cast(offset, ti.f32) - fx) * self.dx
                weight = w[i][0] * w[j][1] * w[k][2]
                idx = base + offset
                if 0 <= idx[0] < self.grid.res and 0 <= idx[1] < self.grid.res and 0 <= idx[2] < self.grid.res:
                    g_v = self.grid.node_vel[idx]
                    new_v += weight * g_v
                    new_C += 4.0 * self.inv_dx * weight * g_v.outer_product(dpos)

            self.particles[p].vel = new_v
            self.particles[p].C = new_C
            self.particles[p].pos += dt * new_v

            # Update F = (I + dt·C) · F  (MLS-MPM update)
            new_F = (ti.Matrix.identity(ti.f32, 3) + dt * new_C) @ self.particles[p].F
            self.particles[p].F = self._return_mapping(new_F, p)

    @ti.func
    def _return_mapping(self, F: ti.types.matrix(3, 3, ti.f32), p: ti.i32) -> ti.types.matrix(3, 3, ti.f32):
        """Drucker-Prager return mapping on F's singular values."""
        U, sig, V = ti.svd(F, ti.f32)
        # Clamp positive (avoid inversion artifacts)
        for d in ti.static(range(3)):
            if sig[d, d] < 0.05:
                sig[d, d] = 0.05
            if sig[d, d] > 20.0:
                sig[d, d] = 20.0

        # log singular values
        e0 = ti.log(sig[0, 0])
        e1 = ti.log(sig[1, 1])
        e2 = ti.log(sig[2, 2])
        tr = e0 + e1 + e2
        # Deviatoric part
        d0 = e0 - tr / 3.0
        d1 = e1 - tr / 3.0
        d2 = e2 - tr / 3.0
        dev_norm = ti.sqrt(d0 * d0 + d1 * d1 + d2 * d2)

        new_e = ti.Vector([e0, e1, e2])
        if tr > 0:
            # Tensile: project to cone tip (Stomakhin-style; simplified)
            new_e = ti.Vector([0.0, 0.0, 0.0])
            self.particles[p].Jp *= ti.exp(tr)
        elif dev_norm > 1e-9:
            mu = self.mu[None]
            la = self.la[None]
            yield_amount = self.alpha[None] * (3.0 * la + 2.0 * mu) / (2.0 * mu) * tr
            new_dev_norm = ti.max(0.0, dev_norm + yield_amount)
            scale = new_dev_norm / dev_norm
            new_e = ti.Vector([d0 * scale + tr / 3.0, d1 * scale + tr / 3.0, d2 * scale + tr / 3.0])
        sig[0, 0] = ti.exp(new_e[0])
        sig[1, 1] = ti.exp(new_e[1])
        sig[2, 2] = ti.exp(new_e[2])

        return U @ sig @ V.transpose()

    def step(self, dt: float, gravity_y: float = -9.8):
        """One MPM substep: clear grid, P2G, grid update, G2P."""
        self.grid.clear()
        self.p2g()
        self.grid.grid_update(dt, ti.Vector([0.0, gravity_y, 0.0]))
        self.g2p(dt)
