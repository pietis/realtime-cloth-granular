"""XPBD (Extended Position-Based Dynamics) cloth solver.

Distance + bending + ground-plane collision constraints. Effective inverse mass
respects per-vertex `mass + Σ_T σ_T·w_vT` (cloth dynamics shifts under reservoir
loading) — this is the *dynamic mass-bearing reservoir* novelty from 계획서.md §4.5.

Material-frame UVs and per-triangle σ live in `triangle.py`. p_σ (reservoir
momentum) lives in `vertex.py`. They are *populated* here (effective mass uses
them) but the actual transfer is in `coupling/attach.py`.
"""

import math

import numpy as np
import taichi as ti


@ti.data_oriented
class ClothSolver:
    """XPBD cloth on a regular grid mesh (rectangular sheet)."""

    def __init__(
        self,
        n_x: int,
        n_y: int,
        size_x: float,
        size_y: float,
        origin: tuple[float, float, float] = (0.0, 1.0, 0.0),
        density: float = 0.3,        # kg / m²
        stiffness_distance: float = 1.0e3,
        stiffness_bending: float = 1.0e2,
        ground_y: float = 0.0,
        n_iterations: int = 8,
    ):
        self.n_x = n_x
        self.n_y = n_y
        self.n_vertices = n_x * n_y
        self.n_triangles = 2 * (n_x - 1) * (n_y - 1)
        self.size_x = size_x
        self.size_y = size_y
        self.origin = origin
        self.density = density
        self.stiffness_distance = stiffness_distance
        self.stiffness_bending = stiffness_bending
        self.ground_y = ground_y
        self.n_iterations = n_iterations

        from .triangle import make_triangle_field
        from .vertex import make_vertex_field

        self.vertices = make_vertex_field(self.n_vertices)
        self.triangles = make_triangle_field(self.n_triangles)

        # Per-vertex tmp for σ aggregation in update_inv_mass
        self.added_sigma = ti.field(ti.f32, shape=self.n_vertices)

        # Distance constraints (edges)
        # Built once at init; stored as flat fields
        edges, rest_lens = self._build_edge_list()
        self.n_edges = len(edges)
        self.edge_v0 = ti.field(ti.i32, shape=self.n_edges)
        self.edge_v1 = ti.field(ti.i32, shape=self.n_edges)
        self.edge_rest = ti.field(ti.f32, shape=self.n_edges)
        for i, (a, b) in enumerate(edges):
            self.edge_v0[i] = a
            self.edge_v1[i] = b
            self.edge_rest[i] = rest_lens[i]

        self._init_geometry()

    # ---- mesh construction ----------------------------------------------------------

    def _build_edge_list(self) -> tuple[list[tuple[int, int]], list[float]]:
        edges = []
        rest_lens = []
        dx = self.size_x / max(self.n_x - 1, 1)
        dy = self.size_y / max(self.n_y - 1, 1)
        for j in range(self.n_y):
            for i in range(self.n_x):
                vi = j * self.n_x + i
                if i + 1 < self.n_x:
                    edges.append((vi, j * self.n_x + (i + 1)))
                    rest_lens.append(dx)
                if j + 1 < self.n_y:
                    edges.append((vi, (j + 1) * self.n_x + i))
                    rest_lens.append(dy)
                # Diagonal (shear) — same triangulation as triangles below
                if i + 1 < self.n_x and j + 1 < self.n_y:
                    edges.append((vi, (j + 1) * self.n_x + (i + 1)))
                    rest_lens.append(math.sqrt(dx * dx + dy * dy))
        return edges, rest_lens

    @ti.kernel
    def _init_vertices_kernel(
        self,
        ox: ti.f32, oy: ti.f32, oz: ti.f32,
        sx: ti.f32, sy: ti.f32,
        n_x: ti.i32, n_y: ti.i32,
        m_per_v: ti.f32,
    ):
        for j, i in ti.ndrange(n_y, n_x):
            v = j * n_x + i
            u = ti.cast(i, ti.f32) / ti.cast(ti.max(n_x - 1, 1), ti.f32)
            t = ti.cast(j, ti.f32) / ti.cast(ti.max(n_y - 1, 1), ti.f32)
            self.vertices[v].pos = ti.Vector([ox + u * sx, oy, oz + t * sy])
            self.vertices[v].vel = ti.Vector([0.0, 0.0, 0.0])
            self.vertices[v].pred = self.vertices[v].pos
            self.vertices[v].mass = m_per_v
            self.vertices[v].inv_mass = 1.0 / m_per_v
            self.vertices[v].p_sigma = ti.Vector([0.0, 0.0, 0.0])
            self.vertices[v].fixed = 0

    def _init_geometry(self):
        m_per_v = self.density * self.size_x * self.size_y / max(self.n_vertices, 1)
        self._init_vertices_kernel(
            self.origin[0], self.origin[1], self.origin[2],
            self.size_x, self.size_y,
            self.n_x, self.n_y,
            m_per_v,
        )

        # Triangulate (two triangles per quad)
        dx = self.size_x / max(self.n_x - 1, 1)
        dy = self.size_y / max(self.n_y - 1, 1)
        rest_area = 0.5 * dx * dy
        idx = 0
        for j in range(self.n_y - 1):
            for i in range(self.n_x - 1):
                v00 = j * self.n_x + i
                v10 = j * self.n_x + (i + 1)
                v01 = (j + 1) * self.n_x + i
                v11 = (j + 1) * self.n_x + (i + 1)

                u_l = i / max(self.n_x - 1, 1)
                u_r = (i + 1) / max(self.n_x - 1, 1)
                t_b = j / max(self.n_y - 1, 1)
                t_t = (j + 1) / max(self.n_y - 1, 1)

                # Triangle 1: v00, v10, v11
                self.triangles[idx].v0 = v00
                self.triangles[idx].v1 = v10
                self.triangles[idx].v2 = v11
                self.triangles[idx].sigma_front = 0.0
                self.triangles[idx].sigma_back = 0.0
                self.triangles[idx].rest_area = rest_area
                self.triangles[idx].uv0 = ti.Vector([u_l, t_b])
                self.triangles[idx].uv1 = ti.Vector([u_r, t_b])
                self.triangles[idx].uv2 = ti.Vector([u_r, t_t])
                idx += 1
                # Triangle 2: v00, v11, v01
                self.triangles[idx].v0 = v00
                self.triangles[idx].v1 = v11
                self.triangles[idx].v2 = v01
                self.triangles[idx].sigma_front = 0.0
                self.triangles[idx].sigma_back = 0.0
                self.triangles[idx].rest_area = rest_area
                self.triangles[idx].uv0 = ti.Vector([u_l, t_b])
                self.triangles[idx].uv1 = ti.Vector([u_r, t_t])
                self.triangles[idx].uv2 = ti.Vector([u_l, t_t])
                idx += 1

    # ---- pin / unpin ----------------------------------------------------------------

    def pin_vertices(self, indices: list[int]):
        idx_field = ti.field(ti.i32, shape=len(indices))
        for k, v in enumerate(indices):
            idx_field[k] = v
        self._pin_kernel(idx_field, len(indices))

    @ti.kernel
    def _pin_kernel(self, idx: ti.template(), n: ti.i32):
        for k in range(n):
            v = idx[k]
            self.vertices[v].fixed = 1
            self.vertices[v].inv_mass = 0.0
            self.vertices[v].vel = ti.Vector([0.0, 0.0, 0.0])

    # ---- XPBD step ------------------------------------------------------------------

    @ti.kernel
    def predict(self, dt: ti.f32, gravity: ti.types.vector(3, ti.f32)):
        for v in range(self.n_vertices):
            if self.vertices[v].fixed == 1:
                self.vertices[v].pred = self.vertices[v].pos
                continue
            # Gravity + reservoir momentum impulse (p_σ already accumulated externally)
            self.vertices[v].vel += dt * gravity
            self.vertices[v].pred = self.vertices[v].pos + dt * self.vertices[v].vel

    @ti.kernel
    def update_inv_mass(self):
        """inv_mass = 1 / (vertex_mass + Σ_T (σ_front + σ_back)·w_vT).

        Two-pass:
          (1) reset added_sigma_per_vertex to 0
          (2) atomic-add per-vertex 1/3 contribution from each triangle's σ
          (3) inv_mass = 1 / (mass + added_sigma)
        """
        for v in range(self.n_vertices):
            self.added_sigma[v] = 0.0
        for t in range(self.n_triangles):
            sigma_total = self.triangles[t].sigma_front + self.triangles[t].sigma_back
            if sigma_total > 1e-12:
                third = sigma_total / 3.0
                ti.atomic_add(self.added_sigma[self.triangles[t].v0], third)
                ti.atomic_add(self.added_sigma[self.triangles[t].v1], third)
                ti.atomic_add(self.added_sigma[self.triangles[t].v2], third)
        for v in range(self.n_vertices):
            if self.vertices[v].fixed == 1:
                self.vertices[v].inv_mass = 0.0
            else:
                m = self.vertices[v].mass + self.added_sigma[v]
                self.vertices[v].inv_mass = 1.0 / ti.max(m, 1e-12)

    @ti.kernel
    def reconcile_velocity_with_p_sigma(self):
        """v_eff = (m·v + p_σ) / (m + σ_total) — absorb reservoir momentum then reset."""
        for v in range(self.n_vertices):
            if self.vertices[v].fixed == 1:
                continue
            m_eff = 1.0 / ti.max(self.vertices[v].inv_mass, 1e-12)
            self.vertices[v].vel = (
                self.vertices[v].mass * self.vertices[v].vel + self.vertices[v].p_sigma
            ) / ti.max(m_eff, 1e-12)
            self.vertices[v].p_sigma = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def solve_distance(self):
        """One Gauss-Seidel pass over distance constraints."""
        for e in range(self.n_edges):
            a = self.edge_v0[e]
            b = self.edge_v1[e]
            wa = self.vertices[a].inv_mass
            wb = self.vertices[b].inv_mass
            w = wa + wb
            if w < 1e-12:
                continue
            d = self.vertices[a].pred - self.vertices[b].pred
            n = d.norm()
            if n < 1e-9:
                continue
            c = n - self.edge_rest[e]
            corr = (c / n) / w * d * self.stiffness_distance / (self.stiffness_distance + 1.0)
            self.vertices[a].pred -= wa * corr
            self.vertices[b].pred += wb * corr

    @ti.kernel
    def solve_ground(self):
        for v in range(self.n_vertices):
            if self.vertices[v].fixed == 1:
                continue
            if self.vertices[v].pred.y < self.ground_y:
                self.vertices[v].pred.y = self.ground_y

    @ti.kernel
    def update(self, dt: ti.f32):
        for v in range(self.n_vertices):
            if self.vertices[v].fixed == 1:
                self.vertices[v].vel = ti.Vector([0.0, 0.0, 0.0])
                continue
            self.vertices[v].vel = (self.vertices[v].pred - self.vertices[v].pos) / dt
            self.vertices[v].pos = self.vertices[v].pred

    def step(self, dt: float, gravity_y: float = -9.8):
        """One full XPBD step (predict → solve → update)."""
        self.update_inv_mass()
        self.reconcile_velocity_with_p_sigma()
        self.predict(dt, ti.Vector([0.0, gravity_y, 0.0]))
        for _ in range(self.n_iterations):
            self.solve_distance()
            self.solve_ground()
        self.update(dt)
