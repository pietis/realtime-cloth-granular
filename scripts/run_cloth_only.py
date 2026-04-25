"""Step 3 — Cloth-only XPBD baseline. No sand.

Pin two top corners and let the cloth sag under gravity. Used to validate
distance constraints + ground collision + reservoir-aware mass update.

Run:
    python3 scripts/run_cloth_only.py [--config data/configs/default_jkr.yaml] [--steps 600]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cloth.xpbd import ClothSolver
from src.utils.config import load_scene_config


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="data/configs/default_jkr.yaml")
    parser.add_argument("--steps", type=int, default=600)
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
    cloth = ClothSolver(
        n_x=cfg.cloth_nx,
        n_y=cfg.cloth_ny,
        size_x=cfg.cloth_size,
        size_y=cfg.cloth_size,
        origin=cfg.cloth_origin,
        density=cfg.cloth_density,
    )

    # Pin top two corners (j = n_y - 1, i = 0 and n_x - 1)
    top_left = (cfg.cloth_ny - 1) * cfg.cloth_nx + 0
    top_right = (cfg.cloth_ny - 1) * cfg.cloth_nx + (cfg.cloth_nx - 1)
    cloth.pin_vertices([top_left, top_right])
    print(f"[run_cloth_only] {cfg.cloth_nx}×{cfg.cloth_ny} cloth, pinned corners: {top_left}, {top_right}")

    sub_dt = cfg.dt / cfg.n_substeps
    for step in range(args.steps):
        for _ in range(cfg.n_substeps):
            cloth.step(sub_dt)
        if step % 50 == 0:
            print(f"  step {step:5d}/{args.steps}")

    print("[OK] cloth-only run complete.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
