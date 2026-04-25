"""Cloth triangle carrying per-triangle reservoir σ (front/back) and material-frame UV.

Material-frame σ avoids swimming under cloth deformation — see 설명서.md Step 3.1.
Front/back separation lets each side accumulate independently.
"""

from __future__ import annotations

import taichi as ti


@ti.dataclass
class Triangle:
    v0:          ti.i32
    v1:          ti.i32
    v2:          ti.i32
    sigma_front: ti.f32
    sigma_back:  ti.f32
    rest_area:   ti.f32
    # Material-frame UV per triangle vertex (rest pose, world-stable)
    uv0:         ti.types.vector(2, ti.f32)
    uv1:         ti.types.vector(2, ti.f32)
    uv2:         ti.types.vector(2, ti.f32)


def make_triangle_field(n: int) -> ti.StructField:
    return Triangle.field(shape=n)


@ti.func
def triangle_normal(triangles: ti.template(), vertices: ti.template(), t: ti.i32):
    """Outward normal of triangle t in current pose."""
    p0 = vertices[triangles[t].v0].pos
    p1 = vertices[triangles[t].v1].pos
    p2 = vertices[triangles[t].v2].pos
    return (p1 - p0).cross(p2 - p0).normalized()


@ti.func
def triangle_centroid(triangles: ti.template(), vertices: ti.template(), t: ti.i32):
    p0 = vertices[triangles[t].v0].pos
    p1 = vertices[triangles[t].v1].pos
    p2 = vertices[triangles[t].v2].pos
    return (p0 + p1 + p2) / 3.0


@ti.func
def triangle_area(triangles: ti.template(), vertices: ti.template(), t: ti.i32) -> ti.f32:
    p0 = vertices[triangles[t].v0].pos
    p1 = vertices[triangles[t].v1].pos
    p2 = vertices[triangles[t].v2].pos
    return 0.5 * (p1 - p0).cross(p2 - p0).norm()


@ti.func
def closest_point_on_triangle(
    triangles: ti.template(),
    vertices: ti.template(),
    t: ti.i32,
    q: ti.types.vector(3, ti.f32),
):
    """Closest point on triangle t to query point q. Returns (closest_point, bary)."""
    p0 = vertices[triangles[t].v0].pos
    p1 = vertices[triangles[t].v1].pos
    p2 = vertices[triangles[t].v2].pos
    e0 = p1 - p0
    e1 = p2 - p0
    d = p0 - q

    a = e0.dot(e0)
    b = e0.dot(e1)
    c = e1.dot(e1)
    dd = e0.dot(d)
    ee = e1.dot(d)

    det = a * c - b * b
    s = b * ee - c * dd
    t_param = b * dd - a * ee

    # Clamp barycentric to triangle interior (simplified; sufficient for proof-of-concept)
    if s + t_param <= det:
        if s < 0.0:
            if t_param < 0.0:
                if dd < 0.0:
                    s = ti.max(0.0, ti.min(1.0, -dd / a))
                    t_param = 0.0
                else:
                    s = 0.0
                    t_param = ti.max(0.0, ti.min(1.0, -ee / c))
            else:
                s = 0.0
                t_param = ti.max(0.0, ti.min(1.0, -ee / c))
        elif t_param < 0.0:
            s = ti.max(0.0, ti.min(1.0, -dd / a))
            t_param = 0.0
        else:
            inv_det = 1.0 / det
            s *= inv_det
            t_param *= inv_det
    else:
        if s < 0.0:
            tmp0 = b + dd
            tmp1 = c + ee
            if tmp1 > tmp0:
                numer = tmp1 - tmp0
                denom = a - 2.0 * b + c
                s = ti.max(0.0, ti.min(1.0, numer / denom))
                t_param = 1.0 - s
            else:
                t_param = ti.max(0.0, ti.min(1.0, -ee / c))
                s = 0.0
        elif t_param < 0.0:
            if a + dd > b + ee:
                numer = c + ee - b - dd
                denom = a - 2.0 * b + c
                s = ti.max(0.0, ti.min(1.0, numer / denom))
                t_param = 1.0 - s
            else:
                s = ti.max(0.0, ti.min(1.0, -ee / c))
                t_param = 0.0
        else:
            numer = c + ee - b - dd
            denom = a - 2.0 * b + c
            s = ti.max(0.0, ti.min(1.0, numer / denom))
            t_param = 1.0 - s

    closest = p0 + s * e0 + t_param * e1
    bary = ti.Vector([1.0 - s - t_param, s, t_param])
    return closest, bary
