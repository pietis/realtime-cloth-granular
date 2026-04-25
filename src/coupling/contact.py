"""Sphere-vs-triangle contact between sand particles and cloth (Step 4).

This is the *contact-only* baseline — particles bounce off triangles, no mass
transfer. Operator A (`attach.py`) replaces the bounce with conditional attach
based on JKR phase-transition criterion.

For broad phase we use a uniform spatial hash on cloth triangle centroids.
For narrow phase we use closest-point-on-triangle from `cloth.triangle`.
"""

from __future__ import annotations

import taichi as ti

from ..cloth.triangle import (
    closest_point_on_triangle,
    triangle_centroid,
    triangle_normal,
)


@ti.data_oriented
class ContactSolver:
    """Sand particle ↔ cloth triangle penalty contact (no transfer)."""

    def __init__(
        self,
        particles: ti.StructField,
        cloth,
        contact_radius: float = 0.02,
        restitution: float = 0.1,
        friction: float = 0.3,
    ):
        self.particles = particles
        self.cloth = cloth
        self.contact_radius = contact_radius
        self.restitution = restitution
        self.friction = friction

    @ti.kernel
    def solve(self):
        """Naive O(N·T) sphere-vs-triangle contact.

        For MVP this is acceptable up to ~50k particles × ~1k triangles.
        Replace with spatial hash in next session for larger scenes.
        """
        for p in range(self.particles.shape[0]):
            if self.particles[p].active == 0:
                continue
            for t in range(self.cloth.n_triangles):
                cp, _bary = closest_point_on_triangle(
                    self.cloth.triangles, self.cloth.vertices, t, self.particles[p].pos
                )
                d_vec = self.particles[p].pos - cp
                d = d_vec.norm()
                threshold = self.contact_radius + self.particles[p].radius
                if d > threshold or d < 1e-9:
                    continue

                n = triangle_normal(self.cloth.triangles, self.cloth.vertices, t)
                # Sign: project particle outward
                if d_vec.dot(n) < 0.0:
                    n = -n
                penetration = threshold - d
                # Project particle out
                self.particles[p].pos += n * penetration
                # Reflect normal velocity (with restitution + friction)
                v = self.particles[p].vel
                vn = v.dot(n)
                if vn < 0.0:
                    v_t = v - vn * n
                    v = v_t * (1.0 - self.friction) - n * vn * self.restitution
                    self.particles[p].vel = v
