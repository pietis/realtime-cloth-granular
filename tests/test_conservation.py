"""Conservation tests for Operator A.

The audit P (계획서 §4.6) is preserved at *substep boundaries* — i.e., after
attach + reconcile, not strictly between them. With v_v = 0 (cloth at rest)
both states coincide. With v_v ≠ 0 the intermediate post-attach state has
a transient drift of m_p · v̄_T_old that reconcile exactly cancels.

We test both regimes to lock down the contract.
"""

import sys
from pathlib import Path

import numpy as np
import pytest
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


@ti.kernel
def _inject_uniform_velocity(
    cloth_vertices: ti.template(), n: ti.i32,
    vx: ti.f32, vy: ti.f32, vz: ti.f32,
):
    for v in range(n):
        cloth_vertices[v].vel = ti.Vector([vx, vy, vz])


def _build_scene():
    from src.cloth.xpbd import ClothSolver
    from src.coupling.attach import AttachOperator
    from src.mpm.particles import init_particles_box, make_particle_field

    n = 4
    particles = make_particle_field(n)
    init_particles_box(
        particles, n,
        ti.Vector([0.45, 0.55, 0.45]),
        ti.Vector([0.55, 0.60, 0.55]),
        mass_per_particle=1.0e-4,
        default_radius=5.0e-3,
    )
    cloth = ClothSolver(
        n_x=2, n_y=2, size_x=0.2, size_y=0.2,
        origin=(0.4, 0.50, 0.4), density=0.3,
    )
    attach_op = AttachOperator(
        particles, cloth,
        gamma_0=10.0, beta=0.0, humidity=0.0, sigma_max=10.0,
        k_reduced=1.0e3, lam=1.0, contact_radius=0.10,
    )
    return particles, cloth, attach_op


def test_attach_preserves_total_mass_and_momentum_at_rest():
    """v_v = 0: attach alone preserves the audit (no reconcile needed)."""
    from src.utils.conservation import total_linear_momentum, total_mass

    ti.init(arch=ti.cpu)
    particles, cloth, attach_op = _build_scene()

    M_before = float(total_mass(particles, cloth.triangles))
    P_before = np.array(total_linear_momentum(particles, cloth.triangles, cloth.vertices))
    attach_op.reset_audit()
    attach_op.step()
    M_after = float(total_mass(particles, cloth.triangles))
    P_after = np.array(total_linear_momentum(particles, cloth.triangles, cloth.vertices))

    n_attached = int(attach_op.attach_event_count[None])
    assert n_attached > 0, "no attach events fired — JKR threshold may be too strict"
    assert abs(M_before - M_after) / M_before < 1e-5, (
        f"mass drift: before={M_before}, after={M_after}, attached={n_attached}"
    )
    p_ref = max(float(np.linalg.norm(P_before)), 1e-12)
    p_err = float(np.linalg.norm(P_after - P_before)) / p_ref
    assert p_err < 1e-5, (
        f"momentum drift at rest: before={P_before}, after={P_after}"
    )


def test_attach_plus_reconcile_preserves_momentum_with_moving_cloth():
    """v_v ≠ 0: only after reconcile is the audit invariant.

    Theorem hypothesis (§4.6): conservation holds at substep boundaries.
    Intermediate post-attach state has a transient `Δ = m_p · v̄_T_old`,
    which `reconcile_velocity_with_p_sigma` cancels exactly.
    """
    from src.utils.conservation import total_linear_momentum, total_mass

    ti.init(arch=ti.cpu)
    particles, cloth, attach_op = _build_scene()

    # Inject a non-trivial uniform velocity onto the cloth.
    _inject_uniform_velocity(cloth.vertices, cloth.n_vertices, 0.7, -0.2, 0.3)

    P_before = np.array(total_linear_momentum(particles, cloth.triangles, cloth.vertices))
    M_before = float(total_mass(particles, cloth.triangles))

    attach_op.reset_audit()
    attach_op.step()                       # attach (no reconcile yet)

    # Intermediate state: audit has drifted by Σ_p m_p · v̄_T_old (= cloth velocity, uniform).
    P_intermediate = np.array(total_linear_momentum(particles, cloth.triangles, cloth.vertices))
    n_attached = int(attach_op.attach_event_count[None])
    expected_drift = n_attached * 1.0e-4 * np.array([0.7, -0.2, 0.3])
    drift_err = np.linalg.norm((P_intermediate - P_before) - expected_drift) / max(np.linalg.norm(expected_drift), 1e-12)
    assert drift_err < 1e-4, (
        f"intermediate drift mismatch: observed={P_intermediate-P_before}, "
        f"expected={expected_drift}, rel_err={drift_err}"
    )

    # Now reconcile — substep boundary.
    cloth._snapshot_prev_sigma()
    cloth.update_inv_mass()
    cloth.reconcile_velocity_with_p_sigma()

    P_after = np.array(total_linear_momentum(particles, cloth.triangles, cloth.vertices))
    M_after = float(total_mass(particles, cloth.triangles))

    assert abs(M_before - M_after) / M_before < 1e-5, (
        f"mass drift across substep: before={M_before}, after={M_after}"
    )
    p_ref = max(float(np.linalg.norm(P_before)), 1e-12)
    p_err = float(np.linalg.norm(P_after - P_before)) / p_ref
    assert p_err < 1e-5, (
        f"momentum NOT conserved at substep boundary: before={P_before}, "
        f"after={P_after}, drift={P_after - P_before}, rel_err={p_err}"
    )
