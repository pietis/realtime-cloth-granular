"""Mass conservation sanity for Operator A — attach event must preserve total mass.

This is a tiny CPU-mode test: 1 active particle, 1 cloth triangle, force-low
JKR threshold so attachment fires deterministically. Total mass before == after.
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_attach_preserves_total_mass():
    import taichi as ti

    from src.cloth.xpbd import ClothSolver
    from src.coupling.attach import AttachOperator
    from src.mpm.particles import init_particles_box, make_particle_field
    from src.utils.conservation import total_mass

    ti.init(arch=ti.cpu)

    # Tiny scene: 4 particles right above a 1-triangle cloth chunk
    n = 4
    particles = make_particle_field(n)
    init_particles_box(
        particles,
        n,
        ti.Vector([0.45, 0.55, 0.45]),
        ti.Vector([0.55, 0.60, 0.55]),
        mass_per_particle=1.0e-4,
        default_radius=5.0e-3,
    )

    # 2x2 cloth (1 quad = 2 triangles) just below particles
    cloth = ClothSolver(
        n_x=2, n_y=2,
        size_x=0.2, size_y=0.2,
        origin=(0.4, 0.50, 0.4),
        density=0.3,
    )

    # Aggressive params so attach fires at first frame regardless of E_kin
    attach_op = AttachOperator(
        particles, cloth,
        gamma_0=10.0,           # huge surface energy → W_adh dominates
        beta=0.0,
        humidity=0.0,
        sigma_max=10.0,         # never saturate
        k_reduced=1.0e3,
        lam=1.0,
        contact_radius=0.10,    # large cutoff so all 4 particles are within
    )

    M_before = float(total_mass(particles, cloth.triangles))
    attach_op.reset_audit()
    attach_op.step()
    M_after = float(total_mass(particles, cloth.triangles))

    # Attach should have happened (4 particles eligible)
    n_attached = int(attach_op.attach_event_count[None])
    assert n_attached > 0, "no attach events fired — JKR threshold may be too strict"

    # Total mass conserved (single-precision tolerance)
    assert abs(M_before - M_after) / M_before < 1e-5, (
        f"mass not conserved: before={M_before}, after={M_after}, attached={n_attached}"
    )
