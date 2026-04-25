"""Sanity: every module imports and Taichi initializes (CPU fallback)."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_taichi_available():
    import taichi as ti  # noqa: F401


def test_imports():
    import taichi as ti

    ti.init(arch=ti.cpu)
    # Each top-level module must import without errors.
    from src.cloth import triangle, vertex, xpbd  # noqa: F401
    from src.coupling import attach, contact, jkr  # noqa: F401
    from src.mpm import grid, particles, sand  # noqa: F401
    from src.utils import config, conservation, visualize  # noqa: F401


def test_config_loads():
    from pathlib import Path as _P

    from src.utils.config import load_scene_config

    repo_root = _P(__file__).resolve().parent.parent
    cfg = load_scene_config(repo_root / "data" / "configs" / "default_jkr.yaml")
    assert cfg.max_particles >= cfg.n_active_particles
    assert cfg.grid_res >= 16
    assert cfg.jkr.gamma_0 > 0
    assert cfg.jkr.k_reduced > 0
