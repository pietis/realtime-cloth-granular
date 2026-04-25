"""Eulerian grid for MPM — node mass / velocity / momentum buffers.

The grid lives in [0, domain_size]^3 with GRID_RES cells per axis.
P2G splatters particle mass+momentum into grid nodes; G2P interpolates back.
We use a 3x3x3 quadratic B-spline kernel (MLS-MPM convention).
"""

import taichi as ti


@ti.data_oriented
class Grid:
    """Eulerian background grid for MPM."""

    def __init__(self, grid_res: int, domain_size: float):
        self.res = grid_res
        self.dx = domain_size / grid_res
        self.inv_dx = 1.0 / self.dx
        self.domain_size = domain_size

        # Node fields (mass, momentum, velocity)
        self.node_mass = ti.field(ti.f32, shape=(grid_res, grid_res, grid_res))
        self.node_mom  = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))
        self.node_vel  = ti.Vector.field(3, ti.f32, shape=(grid_res, grid_res, grid_res))

    @ti.kernel
    def clear(self):
        for I in ti.grouped(self.node_mass):
            self.node_mass[I] = 0.0
            self.node_mom[I] = ti.Vector([0.0, 0.0, 0.0])
            self.node_vel[I] = ti.Vector([0.0, 0.0, 0.0])

    @ti.kernel
    def grid_update(self, dt: ti.f32, gravity: ti.types.vector(3, ti.f32)):
        """Convert momentum to velocity, apply gravity, enforce box boundary."""
        for I in ti.grouped(self.node_mass):
            if self.node_mass[I] > 1e-10:
                v = self.node_mom[I] / self.node_mass[I]
                v += gravity * dt
                # Box boundary (sticky for simplicity)
                for k in ti.static(range(3)):
                    if I[k] < 3 and v[k] < 0.0:
                        v[k] = 0.0
                    if I[k] > self.res - 3 and v[k] > 0.0:
                        v[k] = 0.0
                self.node_vel[I] = v
                self.node_mom[I] = v * self.node_mass[I]
            else:
                self.node_vel[I] = ti.Vector([0.0, 0.0, 0.0])
