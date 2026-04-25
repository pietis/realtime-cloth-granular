"""σ field visualization helpers — per-vertex average for color mapping.

For MVP we just compute per-vertex σ density (kg/m²) as a Python-side numpy
array suitable for matplotlib colormap. Real-time GUI rendering with ti.GUI
is left to script-level code.
"""

import numpy as np
import taichi as ti


@ti.kernel
def aggregate_sigma_to_vertices(
    cloth_vertices: ti.template(),
    cloth_triangles: ti.template(),
    sigma_per_vertex: ti.template(),
):
    """Sum σ_front + σ_back of incident triangles into a per-vertex scalar (1/3 each)."""
    for v in range(cloth_vertices.shape[0]):
        sigma_per_vertex[v] = 0.0
    for t in range(cloth_triangles.shape[0]):
        s = cloth_triangles[t].sigma_front + cloth_triangles[t].sigma_back
        v0 = cloth_triangles[t].v0
        v1 = cloth_triangles[t].v1
        v2 = cloth_triangles[t].v2
        ti.atomic_add(sigma_per_vertex[v0], s / 3.0)
        ti.atomic_add(sigma_per_vertex[v1], s / 3.0)
        ti.atomic_add(sigma_per_vertex[v2], s / 3.0)


def per_vertex_sigma_numpy(cloth) -> np.ndarray:
    """Convenience: aggregate σ to per-vertex and return numpy array.

    Allocates a fresh scratch field each call — caching breaks if ti.init() is
    called again between calls (e.g. parameter sweeps), and Taichi alloc cost
    is negligible compared to the aggregation kernel.
    """
    buf = ti.field(ti.f32, shape=cloth.n_vertices)
    aggregate_sigma_to_vertices(cloth.vertices, cloth.triangles, buf)
    return buf.to_numpy()
