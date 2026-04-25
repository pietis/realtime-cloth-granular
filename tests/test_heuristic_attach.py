"""B1 HeuristicAttachOperator — sanity tests.

Tests:
  1. Mass conservation: aggressive p_0=1.0, v_th=10.0 forces all eligible
     particles to attach; total mass (particles + cloth σ) must be conserved
     to single-precision tolerance (rel-err < 1e-5).

Mirror of tests/test_conservation.py structure, adapted for B1 baseline.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_heuristic_attach_mass_conservation():
    """HeuristicAttachOperator with p_0=1.0, v_th=10 must conserve total mass."""
    import taichi as ti

    from src.cloth.xpbd import ClothSolver
    from src.coupling.attach_heuristic import HeuristicAttachOperator
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

    # Aggressive params: p_0=1.0, v_th=10.0 → all slow-enough particles always attach
    attach_op = HeuristicAttachOperator(
        particles, cloth,
        sigma_max=10.0,       # never saturate
        lam=1.0,              # large falloff → prob stays ~p_0 over small d
        contact_radius=0.10,  # large cutoff so all 4 particles in range
        p_0=1.0,              # probability = 1.0 (modulo distance factor)
        v_th=10.0,            # v_th very high → velocity gate always passes
    )

    M_before = float(total_mass(particles, cloth.triangles))
    attach_op.reset_audit()
    attach_op.step()
    M_after = float(total_mass(particles, cloth.triangles))

    # Attach should have happened (4 particles eligible with p_0=1.0)
    n_attached = int(attach_op.attach_event_count[None])
    assert n_attached > 0, (
        "no attach events fired — check p_0=1.0 + v_th=10.0 path in HeuristicAttachOperator"
    )

    # Total mass conserved to single-precision tolerance
    assert abs(M_before - M_after) / M_before < 1e-5, (
        f"mass not conserved: before={M_before:.6e}, after={M_after:.6e}, "
        f"rel_err={abs(M_before - M_after) / M_before:.2e}, attached={n_attached}"
    )
