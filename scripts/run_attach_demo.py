"""Step 5 — JKR phase-transition attach demo (Operator A).

Sand falls onto a horizontal cloth pinned at four corners, sticks per JKR
criterion, accumulates as σ. Audit logs total mass+momentum for conservation
verification.

Run:
    python3 scripts/run_attach_demo.py [--config data/configs/demo_a_lying.yaml] [--out results/attach_demo_log.npz]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.cloth.xpbd import ClothSolver
from src.coupling.attach import AttachOperator
from src.coupling.contact import ContactSolver
from src.mpm.grid import Grid
from src.mpm.particles import (
    count_active,
    init_particles_box,
    make_particle_field,
)
from src.mpm.sand import SandSolver
from src.utils.config import load_scene_config
from src.utils.conservation import (
    relative_error,
    total_linear_momentum,
    total_mass,
)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="data/configs/demo_a_lying.yaml")
    parser.add_argument("--out", type=str, default="results/attach_demo_log.npz")
    parser.add_argument("--steps", type=int, default=None)
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
    n_steps = args.steps if args.steps is not None else int(cfg.duration_seconds / cfg.dt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # Sand setup
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

    # Cloth setup
    cloth = ClothSolver(
        n_x=cfg.cloth_nx,
        n_y=cfg.cloth_ny,
        size_x=cfg.cloth_size,
        size_y=cfg.cloth_size,
        origin=cfg.cloth_origin,
        density=cfg.cloth_density,
    )
    if cfg.pinned_corners and any(cfg.pinned_corners):
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

    # Coupling — contact + Operator A attach
    contact = ContactSolver(particles, cloth, contact_radius=cfg.jkr.contact_radius)
    attach = AttachOperator(
        particles,
        cloth,
        gamma_0=cfg.jkr.gamma_0,
        beta=cfg.jkr.beta,
        humidity=cfg.jkr.humidity,
        sigma_max=cfg.jkr.sigma_max,
        k_reduced=cfg.jkr.k_reduced,
        lam=cfg.jkr.lam,
        contact_radius=cfg.jkr.contact_radius,
    )

    M0 = total_mass(particles, cloth.triangles)
    P0 = total_linear_momentum(particles, cloth.vertices)
    print(f"[attach_demo] initial mass: {M0:.6e} kg, momentum: {P0[0]:+.4e}, {P0[1]:+.4e}, {P0[2]:+.4e}")
    print(f"  steps: {n_steps}, dt: {cfg.dt}, substeps: {cfg.n_substeps}, humidity: {cfg.jkr.humidity}")

    log = {
        "step": [],
        "n_active": [],
        "n_attached_total": [],
        "mass_drift": [],
        "momentum_drift_norm": [],
        "attach_events_per_step": [],
    }

    n_attached_cum = 0
    sub_dt = cfg.dt / cfg.n_substeps
    for step in range(n_steps):
        attach.reset_audit()
        for _ in range(cfg.n_substeps):
            sand.step(sub_dt)
            cloth.step(sub_dt)
            contact.solve()
            attach.step()
        n_attached_cum += attach.attach_event_count[None]

        if step % 25 == 0:
            n_a = count_active(particles)
            M_now = total_mass(particles, cloth.triangles)
            P_now = total_linear_momentum(particles, cloth.vertices)
            err_M = relative_error(M_now, M0)
            err_P = float((P_now - P0).norm() / max(P0.norm(), 1.0))
            log["step"].append(step)
            log["n_active"].append(int(n_a))
            log["n_attached_total"].append(int(n_attached_cum))
            log["mass_drift"].append(err_M)
            log["momentum_drift_norm"].append(err_P)
            log["attach_events_per_step"].append(int(attach.attach_event_count[None]))
            print(
                f"  step {step:5d}/{n_steps}  active={n_a:5d}  attached_cum={n_attached_cum:5d}  "
                f"|ΔM|/M0={err_M:.2e}  events_this_substep={attach.attach_event_count[None]}"
            )

    # Save final log + per-vertex σ snapshot
    from src.utils.visualize import per_vertex_sigma_numpy

    sigma_snapshot = per_vertex_sigma_numpy(cloth)
    np.savez(
        out_path,
        step=np.array(log["step"]),
        n_active=np.array(log["n_active"]),
        n_attached_total=np.array(log["n_attached_total"]),
        mass_drift=np.array(log["mass_drift"]),
        momentum_drift_norm=np.array(log["momentum_drift_norm"]),
        attach_events_per_step=np.array(log["attach_events_per_step"]),
        sigma_per_vertex=sigma_snapshot,
        cloth_n_x=cfg.cloth_nx,
        cloth_n_y=cfg.cloth_ny,
    )
    print(f"[OK] log saved to {out_path}")
    print(f"  total attached: {n_attached_cum}, sigma min/max/mean: "
          f"{sigma_snapshot.min():.4e} / {sigma_snapshot.max():.4e} / {sigma_snapshot.mean():.4e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
