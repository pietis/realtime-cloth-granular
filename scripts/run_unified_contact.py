"""Step 4 — Sand + cloth, contact-only (no transfer yet).

Sand falls onto the cloth and bounces / piles. Used as the *Dynamic Duo-style*
baseline before Operator A is enabled in `run_attach_demo.py`.

Run:
    python3 scripts/run_unified_contact.py [--config data/configs/demo_a_lying.yaml]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cloth.xpbd import ClothSolver
from src.coupling.contact import ContactSolver
from src.mpm.grid import Grid
from src.mpm.particles import (
    count_active,
    init_particles_box,
    make_particle_field,
)
from src.mpm.sand import SandSolver
from src.utils.config import load_scene_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="data/configs/demo_a_lying.yaml")
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    if args.cpu:
        ti.init(arch=ti.cpu)
    else:
        try:
            ti.init(arch=ti.cuda)
        except Exception:
            ti.init(arch=ti.cpu)

    cfg = load_scene_config(args.config)

    # Sand
    particles = make_particle_field(cfg.max_particles)
    init_particles_box(
        particles,
        cfg.n_active_particles,
        ti.Vector(cfg.sand_box_min),
        ti.Vector(cfg.sand_box_max),
        cfg.mass_per_particle,
        cfg.particle_radius,
    )
    grid = Grid(cfg.grid_res, cfg.domain_size)
    sand = SandSolver(particles, grid)

    # Cloth
    cloth = ClothSolver(
        n_x=cfg.cloth_nx,
        n_y=cfg.cloth_ny,
        size_x=cfg.cloth_size,
        size_y=cfg.cloth_size,
        origin=cfg.cloth_origin,
        density=cfg.cloth_density,
    )
    if cfg.pinned_corners and any(cfg.pinned_corners):
        # Convention: order = [bottom-left, bottom-right, top-left, top-right]
        bl, br, tl, tr = cfg.pinned_corners
        n_x = cfg.cloth_nx
        n_y = cfg.cloth_ny
        pin = []
        if bl: pin.append(0)
        if br: pin.append(n_x - 1)
        if tl: pin.append((n_y - 1) * n_x)
        if tr: pin.append((n_y - 1) * n_x + (n_x - 1))
        if pin:
            cloth.pin_vertices(pin)
            print(f"[unified_contact] pinned cloth corners: {pin}")

    contact = ContactSolver(particles, cloth, contact_radius=cfg.jkr.contact_radius)

    sub_dt = cfg.dt / cfg.n_substeps
    for step in range(args.steps):
        for _ in range(cfg.n_substeps):
            sand.step(sub_dt)
            cloth.step(sub_dt)
            contact.solve()
        if step % 50 == 0:
            print(f"  step {step:5d}/{args.steps}: active sand = {count_active(particles)}")

    print("[OK] unified contact run complete (no transfer).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
