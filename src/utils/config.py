"""YAML config loader for scene + JKR parameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class JKRConfig:
    gamma_0: float = 0.05
    beta: float = 0.10
    humidity: float = 0.0
    sigma_max: float = 0.05
    k_reduced: float = 1.0e6
    lam: float = 0.005
    contact_radius: float = 0.015


@dataclass
class SceneConfig:
    max_particles: int = 50_000
    n_active_particles: int = 5_000
    grid_res: int = 64
    domain_size: float = 1.0
    particle_radius: float = 0.005
    mass_per_particle: float = 1.0e-4
    cloth_nx: int = 32
    cloth_ny: int = 32
    cloth_size: float = 0.5
    cloth_origin: tuple[float, float, float] = (0.25, 0.6, 0.25)
    cloth_density: float = 0.3
    sand_box_min: tuple[float, float, float] = (0.3, 0.05, 0.3)
    sand_box_max: tuple[float, float, float] = (0.7, 0.2, 0.7)
    dt: float = 1.0e-3
    n_substeps: int = 4
    duration_seconds: float = 30.0
    pinned_corners: tuple[bool, bool, bool, bool] = (False, False, False, False)
    jkr: JKRConfig = field(default_factory=JKRConfig)


def load_scene_config(path: str | Path) -> SceneConfig:
    """Load a YAML config file into SceneConfig (with JKRConfig nested)."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config not found: {p}")
    with p.open("r", encoding="utf-8") as f:
        data: dict[str, Any] = yaml.safe_load(f) or {}

    jkr_data = data.pop("jkr", {}) or {}
    jkr = JKRConfig(**jkr_data)

    # Convert list → tuple for fields typed as tuple
    for k in ("cloth_origin", "sand_box_min", "sand_box_max", "pinned_corners"):
        if k in data and isinstance(data[k], list):
            data[k] = tuple(data[k])

    return SceneConfig(jkr=jkr, **data)
