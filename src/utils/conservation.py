"""Conservation audit — mass and linear momentum across MPM and cloth domains.

For Step 9 we will add a full theorem proof + double-precision audit. For MVP
we use single-precision, which is enough to detect 1% drift over 10k frames.
"""

import taichi as ti


@ti.kernel
def total_mass(particles: ti.template(), cloth_triangles: ti.template()) -> ti.f64:
    total = 0.0
    for p in range(particles.shape[0]):
        if particles[p].active == 1:
            ti.atomic_add(total, ti.cast(particles[p].mass, ti.f64))
    for t in range(cloth_triangles.shape[0]):
        ti.atomic_add(
            total,
            ti.cast(cloth_triangles[t].sigma_front + cloth_triangles[t].sigma_back, ti.f64),
        )
    return total


@ti.kernel
def total_linear_momentum(
    particles: ti.template(), cloth_triangles: ti.template(), cloth_vertices: ti.template()
) -> ti.types.vector(3, ti.f64):
    total = ti.Vector([0.0, 0.0, 0.0], dt=ti.f64)
    for p in range(particles.shape[0]):
        if particles[p].active == 1:
            ti.atomic_add(total, ti.cast(particles[p].mass * particles[p].vel, ti.f64))
    for t in range(cloth_triangles.shape[0]):
        sigma_total = cloth_triangles[t].sigma_front + cloth_triangles[t].sigma_back
        v_bar = (
            cloth_vertices[cloth_triangles[t].v0].vel
            + cloth_vertices[cloth_triangles[t].v1].vel
            + cloth_vertices[cloth_triangles[t].v2].vel
        ) / 3.0
        ti.atomic_add(total, ti.cast(sigma_total * v_bar, ti.f64))
    for v in range(cloth_vertices.shape[0]):
        ti.atomic_add(
            total,
            ti.cast(
                cloth_vertices[v].mass * cloth_vertices[v].vel + cloth_vertices[v].p_sigma,
                ti.f64,
            ),
        )
    return total


def relative_error(current: float, reference: float) -> float:
    if abs(reference) < 1e-12:
        return abs(current - reference)
    return abs(current - reference) / abs(reference)
