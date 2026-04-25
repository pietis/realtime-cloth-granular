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


# Pre-allocated scratch buffer keyed by (n_vertices) so we don't recreate fields.
_sigma_buf_cache: dict[int, "ti.field"] = {}


def per_vertex_sigma_numpy(cloth) -> np.ndarray:
    """Convenience: aggregate σ to per-vertex and return numpy array."""
    n = cloth.n_vertices
    if n not in _sigma_buf_cache:
        _sigma_buf_cache[n] = ti.field(ti.f32, shape=n)
    buf = _sigma_buf_cache[n]
    aggregate_sigma_to_vertices(cloth.vertices, cloth.triangles, buf)
    return buf.to_numpy()
