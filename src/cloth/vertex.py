"""Cloth vertex with reservoir momentum p_σ."""

import taichi as ti


@ti.dataclass
class Vertex:
    pos:      ti.types.vector(3, ti.f32)
    vel:      ti.types.vector(3, ti.f32)
    pred:     ti.types.vector(3, ti.f32)   # predicted position (XPBD predict)
    mass:     ti.f32
    inv_mass: ti.f32
    p_sigma:  ti.types.vector(3, ti.f32)   # reservoir momentum from absorbed particles
    fixed:    ti.i32


def make_vertex_field(n: int) -> ti.StructField:
    return Vertex.field(shape=n)
