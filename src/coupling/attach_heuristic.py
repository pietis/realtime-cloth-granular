"""B1 Baseline — Heuristic stick operator (ablation).

Original W2 (deprecated) heuristic criterion:

    P_stick(i) = p_0 · 1[v_n < v_th] · exp(-d / λ)

That is: a stick *probability* with a velocity-threshold gate and exponential
distance falloff, but **no JKR pull-off energy**.  Compare to Operator A
(attach.py) which uses the energy criterion:

    Attach(i,T) iff E_kin_n < W_JKR · exp(-d/λ)

This baseline exists purely for ablation (계획서.md §7, B1).  It is not
scientifically motivated — the point is that the JKR version produces
qualitatively different (better) behavior in demo (b)(c)(d).

API mirrors AttachOperator exactly so the call site (run_attach_demo.py,
run_ablation_b1.py) can swap operators with a single import change.

Constraints (Taichi 1.7.4 compatibility):
- No `from __future__ import annotations`
- @ti.func parameters must be untyped (Taichi infers)
"""

import taichi as ti

from ..cloth.triangle import closest_point_on_triangle, triangle_normal


@ti.data_oriented
class HeuristicAttachOperator:
    """B1 heuristic-stick attach operator (ablation baseline).

    Parameters mirror AttachOperator.__init__ for call-site compatibility.
    JKR-specific parameters (gamma_0, beta, k_reduced) are accepted but
    ignored internally.

    New heuristic parameters:
        p_0:   base attachment probability [0, 1]  (default 0.3)
        v_th:  normal-velocity threshold (m/s)     (default 0.5)

    Stick criterion:
        draw r ~ Uniform[0,1); attach iff r < p_0 * exp(-d/λ) AND v_n < v_th
    """

    def __init__(
        self,
        particles: ti.StructField,
        cloth,
        # --- JKR params (accepted for call-site compat, ignored internally) ---
        gamma_0: float = 0.05,
        beta: float = 0.10,
        humidity: float = 0.0,
        sigma_max: float = 0.05,
        k_reduced: float = 1.0e6,
        lam: float = 0.005,
        contact_radius: float = 0.015,
        # --- Heuristic-specific params ---
        p_0: float = 0.3,
        v_th: float = 0.5,
    ):
        self.particles = particles
        self.cloth = cloth
        # JKR compat fields (kept so external callers can read them without error)
        self.gamma_0 = gamma_0
        self.beta = beta
        self.humidity = humidity
        self.sigma_max = sigma_max
        self.k_reduced = k_reduced
        self.lam = lam
        self.contact_radius = contact_radius
        # Heuristic params stored as Taichi scalars for kernel access
        self.p_0 = p_0
        self.v_th = v_th

        # Audit counters — same fields/names as AttachOperator
        self.mass_attached_this_frame = ti.field(ti.f32, shape=())
        self.momentum_attached_this_frame = ti.Vector.field(3, ti.f32, shape=())
        self.attach_event_count = ti.field(ti.i32, shape=())

    @ti.kernel
    def reset_audit(self):
        self.mass_attached_this_frame[None] = 0.0
        self.momentum_attached_this_frame[None] = ti.Vector([0.0, 0.0, 0.0])
        self.attach_event_count[None] = 0

    @ti.kernel
    def step(self):
        """One heuristic attach pass — propose+commit fused (same structure as AttachOperator)."""
        for p in range(self.particles.shape[0]):
            if self.particles[p].active == 0:
                continue

            # Find best (closest) triangle within cutoff — identical to AttachOperator
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

            # Heuristic criterion: v_n gate + probabilistic threshold
            n_t = triangle_normal(self.cloth.triangles, self.cloth.vertices, best_t)
            if best_side == -1:
                n_t = -n_t
            v_n = self.particles[p].vel.dot(n_t)
            # Normal inbound speed (positive = approaching)
            v_n_in = ti.max(0.0, -v_n)

            # Velocity gate: particle must be approaching slowly enough
            if v_n_in >= self.v_th:
                continue

            # Probabilistic gate: r < p_0 * exp(-d/λ)
            prob = self.p_0 * ti.exp(-best_d / ti.max(self.lam, 1.0e-9))
            r = ti.random(ti.f32)
            if r >= prob:
                continue

            # σ saturation gate — keep same guard as AttachOperator
            sigma_local = self.cloth.triangles[best_t].sigma_front
            if best_side == -1:
                sigma_local = self.cloth.triangles[best_t].sigma_back
            if sigma_local >= self.sigma_max:
                continue

            # Commit attach — identical bookkeeping to AttachOperator
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
            ti.atomic_add(self.cloth.vertices[v0].p_sigma, best_bary[0] * m * v)
            ti.atomic_add(self.cloth.vertices[v1].p_sigma, best_bary[1] * m * v)
            ti.atomic_add(self.cloth.vertices[v2].p_sigma, best_bary[2] * m * v)
            # 4. Audit
            ti.atomic_add(self.mass_attached_this_frame[None], m)
            ti.atomic_add(self.momentum_attached_this_frame[None], m * v)
            ti.atomic_add(self.attach_event_count[None], 1)
