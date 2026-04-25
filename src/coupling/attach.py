"""Operator A — JKR phase-transition attach (Step 5, core novelty).

The criterion (계획서.md §4.2):

    Attach(i, T) ⇔ E_kin_n^i < W_adh^JKR(R_i, γ_T(σ, h)) · exp(-d / λ)

where E_kin_n is normal-direction kinetic energy at impact.

Implementation pattern: *propose-then-commit*. Each substep:
  1. Each active particle near a triangle proposes an attach event.
  2. Events are committed (particle deactivated, σ_T += m, p_σ_v += m·v / 3).
  3. Audit counters track total mass+momentum transferred.

Sort-then-commit deterministic ordering. p_sigma uses uniform 1/3 to match
update_inv_mass; this preserves global momentum and prevents the
spatial-distribution mismatch with sigma effective-mass.
"""

import taichi as ti

from ..cloth.triangle import closest_point_on_triangle, triangle_normal
from .jkr import gamma_humid, jkr_pulloff_work


@ti.data_oriented
class AttachOperator:
    """JKR-informed attach (Operator A).

    Parameters mirror 계획서.md §4.2:
        gamma_0:    base surface energy γ_0 (J/m²)
        beta:       capillary slope (J/m²) — γ = γ_0 + β·h·(1 - σ/σ_max)
        humidity:   h ∈ [0, 1]
        sigma_max:  saturation cap per triangle (kg)
        k_reduced:  combined modulus K (Pa)
        lam:        falloff length λ (m) in distance term
        contact_radius: cutoff for proximity test (m)
    """

    def __init__(
        self,
        particles: ti.StructField,
        cloth,
        gamma_0: float = 0.05,
        beta: float = 0.10,
        humidity: float = 0.0,
        sigma_max: float = 0.05,
        k_reduced: float = 1.0e6,
        lam: float = 0.005,
        contact_radius: float = 0.015,
    ):
        self.particles = particles
        self.cloth = cloth
        self.gamma_0 = gamma_0
        self.beta = beta
        self.humidity = humidity
        self.sigma_max = sigma_max
        self.k_reduced = k_reduced
        self.lam = lam
        self.contact_radius = contact_radius

        # Audit counters (kg, kg·m/s)
        self.mass_attached_this_frame = ti.field(ti.f32, shape=())
        self.momentum_attached_this_frame = ti.Vector.field(3, ti.f32, shape=())
        self.attach_event_count = ti.field(ti.i32, shape=())
        # Diagnostic: how many particles were close-enough candidates vs how
        # many actually committed an attach. The ratio commits/candidates tells
        # whether JKR is the rate-limiter (low ratio) or contact rate is
        # (high ratio, "lucky impact" regime).
        self.candidate_count = ti.field(ti.i32, shape=())
        self.kin_energy_avg = ti.field(ti.f32, shape=())   # avg of E_kin_n over candidates this frame
        self.threshold_avg = ti.field(ti.f32, shape=())    # avg of W_adh·exp(-d/λ) over candidates

        n_particles = particles.shape[0]
        self.event_active = ti.field(ti.i32, shape=n_particles)
        self.event_triangle = ti.field(ti.i32, shape=n_particles)
        self.event_bary = ti.Vector.field(3, ti.f32, shape=n_particles)
        self.event_side = ti.field(ti.i32, shape=n_particles)

    @ti.kernel
    def reset_audit(self):
        self.mass_attached_this_frame[None] = 0.0
        self.momentum_attached_this_frame[None] = ti.Vector([0.0, 0.0, 0.0])
        self.attach_event_count[None] = 0
        self.candidate_count[None] = 0
        self.kin_energy_avg[None] = 0.0
        self.threshold_avg[None] = 0.0

    @ti.kernel
    def _propose(self):
        """Find attach candidates in parallel."""
        for p in range(self.particles.shape[0]):
            self.event_active[p] = 0
            if self.particles[p].active == 0:
                continue

            # Find best (closest) triangle within cutoff
            best_t = -1
            best_d = 1.0e9
            best_bary = ti.Vector([0.0, 0.0, 0.0])
            best_side = 1   # +1 = front, -1 = back
            for t in range(self.cloth.n_triangles):
                cp, bary = closest_point_on_triangle(
                    self.cloth.triangles, self.cloth.vertices, t, self.particles[p].pos
                )
                diff = self.particles[p].pos - cp
                d = diff.norm()
                if d < best_d and d <= self.contact_radius + self.particles[p].radius:
                    best_d = d
                    best_t = t
                    best_bary = bary
                    n = triangle_normal(self.cloth.triangles, self.cloth.vertices, t)
                    best_side = 1 if diff.dot(n) >= 0.0 else -1

            if best_t < 0:
                continue

            # JKR phase-transition criterion
            n_t = triangle_normal(self.cloth.triangles, self.cloth.vertices, best_t)
            if best_side == -1:
                n_t = -n_t
            v_n = self.particles[p].vel.dot(n_t)
            # Only impacts (incoming particles): take normal-inbound speed
            v_n_in = ti.max(0.0, -v_n)
            E_kin_n = 0.5 * self.particles[p].mass * v_n_in * v_n_in

            sigma_local = self.cloth.triangles[best_t].sigma_front
            if best_side == -1:
                sigma_local = self.cloth.triangles[best_t].sigma_back
            gamma_T = gamma_humid(self.gamma_0, self.beta, self.humidity, sigma_local, self.sigma_max)
            W_adh = jkr_pulloff_work(self.particles[p].radius, gamma_T, self.k_reduced)
            # Falloff uses surface-to-surface gap, not center distance.
            # For particle radius R, gap = max(0, best_d - R). At contact gap≈0,
            # so exp(-gap/λ) ≈ 1 and W_adh dominates. Without this, larger R
            # particles register at far center-distances and exp(-d/λ) wipes
            # out the R^(7/3) advantage from W_adh.
            gap = ti.max(0.0, best_d - self.particles[p].radius)
            threshold = W_adh * ti.exp(-gap / ti.max(self.lam, 1e-9))

            # Diagnostic: this particle was a *candidate* — within contact distance
            # AND with sigma_local headroom. Tracks "JKR-rate vs contact-rate"
            # regime (low commit/candidate ⇒ JKR-dominant, high ⇒ contact-rate).
            if sigma_local < self.sigma_max:
                ti.atomic_add(self.candidate_count[None], 1)
                ti.atomic_add(self.kin_energy_avg[None], E_kin_n)
                ti.atomic_add(self.threshold_avg[None], threshold)

            if E_kin_n < threshold and sigma_local < self.sigma_max:
                self.event_active[p] = 1
                self.event_triangle[p] = best_t
                self.event_bary[p] = best_bary
                self.event_side[p] = best_side

    @ti.kernel
    def _commit(self):
        """Commit proposed events in deterministic particle-index order."""
        ti.loop_config(serialize=True)
        for p in range(self.particles.shape[0]):
            if self.event_active[p] == 0:
                continue

            best_t = self.event_triangle[p]
            best_side = self.event_side[p]

            sigma_local = self.cloth.triangles[best_t].sigma_front
            if best_side == -1:
                sigma_local = self.cloth.triangles[best_t].sigma_back
            if sigma_local >= self.sigma_max:
                continue

            if self.particles[p].active != 0:
                m = self.particles[p].mass
                v = self.particles[p].vel
                # 1. Deactivate particle
                self.particles[p].active = 0
                # 2. σ accumulation (front or back)
                if best_side == 1:
                    ti.atomic_add(self.cloth.triangles[best_t].sigma_front, m)
                else:
                    ti.atomic_add(self.cloth.triangles[best_t].sigma_back, m)
                # 3. p_σ accumulation (3 vertices, barycentric)
                v0 = self.cloth.triangles[best_t].v0
                v1 = self.cloth.triangles[best_t].v1
                v2 = self.cloth.triangles[best_t].v2
                p_share = (1.0 / 3.0) * m * v
                ti.atomic_add(self.cloth.vertices[v0].p_sigma, p_share)
                ti.atomic_add(self.cloth.vertices[v1].p_sigma, p_share)
                ti.atomic_add(self.cloth.vertices[v2].p_sigma, p_share)
                # 4. Audit
                ti.atomic_add(self.mass_attached_this_frame[None], m)
                ti.atomic_add(self.momentum_attached_this_frame[None], m * v)
                ti.atomic_add(self.attach_event_count[None], 1)

    def step(self):
        """One attach pass with parallel proposal and serial commit."""
        self._propose()
        self._commit()
