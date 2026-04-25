"""Step 2 — Sand-only MPM baseline. No cloth, no transfer.

Sand particles fall in a box and pile up. Used to validate MPM solver alone.

Run:
    python3 scripts/run_sand_only.py [--config data/configs/default_jkr.yaml] [--steps 600]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import taichi as ti

# Allow running from project root
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.mpm.grid import Grid
from src.mpm.particles import init_particles_box, make_particle_field
from src.mpm.sand import SandSolver
from src.utils.config import load_scene_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="data/configs/default_jkr.yaml")
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--cpu", action="store_true", help="Force CPU backend")
    args = parser.parse_args()

    if args.cpu:
        ti.init(arch=ti.cpu)
    else:
        try:
            ti.init(arch=ti.cuda)
        except Exception:
            ti.init(arch=ti.cpu)

    cfg = load_scene_config(args.config)
    print(f"[run_sand_only] config: {args.config}")
    print(f"  active sand particles: {cfg.n_active_particles} / cap {cfg.max_particles}")
    print(f"  grid: {cfg.grid_res}^3, dx = {cfg.domain_size / cfg.grid_res:.4f}")

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

    sub_dt = cfg.dt / cfg.n_substeps
    for step in range(args.steps):
        for _ in range(cfg.n_substeps):
            sand.step(sub_dt)
        if step % 50 == 0:
            from src.mpm.particles import count_active

            n_active = count_active(particles)
            print(f"  step {step:5d}/{args.steps}: active = {n_active}")

    print("[OK] sand-only run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
