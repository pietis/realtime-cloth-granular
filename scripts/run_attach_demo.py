"""Step 5 — JKR phase-transition attach demo (Operator A) + B1 baseline.

Sand falls onto a horizontal cloth pinned at four corners, sticks per JKR
criterion (or heuristic, controlled by --baseline), accumulates as σ.
Audit logs total mass+momentum for conservation verification.

Run:
    python3 scripts/run_attach_demo.py [--config data/configs/demo_a_lying.yaml] [--out results/attach_demo_log.npz]
    python3 scripts/run_attach_demo.py --baseline heuristic   # B1 ablation baseline
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
from src.coupling.attach_heuristic import HeuristicAttachOperator
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
    parser.add_argument(
        "--baseline",
        choices=["jkr", "heuristic"],
        default="jkr",
        help="Operator to use: 'jkr' (default, Operator A) or 'heuristic' (B1 ablation)",
    )
    parser.add_argument("--seed", type=int, default=42,
                        help="Taichi random seed (controls particle init RNG)")
    parser.add_argument("--lam", type=float, default=None,
                        help="Override JKR distance falloff λ (m). Set very large (e.g. 1.0) "
                             "to disable distance penalty and isolate W_adh effect.")
    args = parser.parse_args()

    if args.cpu:
        ti.init(arch=ti.cpu, random_seed=args.seed)
    else:
        try:
            ti.init(arch=ti.cuda, random_seed=args.seed)
        except Exception:
            ti.init(arch=ti.cpu, random_seed=args.seed)

    cfg = load_scene_config(args.config)
    if args.lam is not None:
        cfg.jkr.lam = args.lam
    n_steps = args.steps if args.steps is not None else int(cfg.duration_seconds / cfg.dt)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[attach_demo] seed={args.seed}, λ={cfg.jkr.lam}")

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

    # Coupling — contact + attach operator (JKR or heuristic B1 baseline)
    contact = ContactSolver(particles, cloth, contact_radius=cfg.jkr.contact_radius)
    if args.baseline == "heuristic":
        attach = HeuristicAttachOperator(
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
        print("[attach_demo] Using B1 heuristic-stick baseline operator")
    else:
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
        print("[attach_demo] Using JKR Operator A (default)")

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
        "candidates_per_step": [],
        "commit_ratio": [],
        "kin_energy_avg": [],
        "threshold_avg": [],
    }

    n_attached_cum = 0
    n_candidates_cum = 0
    sub_dt = cfg.dt / cfg.n_substeps
    has_diag = hasattr(attach, "candidate_count")
    for step in range(n_steps):
        attach.reset_audit()
        for _ in range(cfg.n_substeps):
            sand.step(sub_dt)
            cloth.step(sub_dt)
            # Order matters: attach must read PRE-contact velocity.
            # Otherwise contact.solve() bounces particles to near-zero v_n,
            # and attach sees E_kin ≈ 0 ⇒ JKR threshold is trivially passed
            # (every candidate commits, "lucky impact" pseudo-regime).
            attach.step()           # JKR phase-transition decides first
            contact.solve()         # remaining (non-attached) particles bounce
        n_attached_cum += attach.attach_event_count[None]
        if has_diag:
            n_candidates_cum += attach.candidate_count[None]

        if step % 25 == 0:
            n_a = count_active(particles)
            M_now = total_mass(particles, cloth.triangles)
            P_now = total_linear_momentum(particles, cloth.vertices)
            err_M = relative_error(M_now, M0)
            err_P = float((P_now - P0).norm() / max(P0.norm(), 1.0))
            cand = int(attach.candidate_count[None]) if has_diag else 0
            ratio = (attach.attach_event_count[None] / max(cand, 1)) if has_diag else 0.0
            kin_avg = float(attach.kin_energy_avg[None] / max(cand, 1)) if has_diag else 0.0
            thr_avg = float(attach.threshold_avg[None] / max(cand, 1)) if has_diag else 0.0
            log["step"].append(step)
            log["n_active"].append(int(n_a))
            log["n_attached_total"].append(int(n_attached_cum))
            log["mass_drift"].append(err_M)
            log["momentum_drift_norm"].append(err_P)
            log["attach_events_per_step"].append(int(attach.attach_event_count[None]))
            log["candidates_per_step"].append(cand)
            log["commit_ratio"].append(ratio)
            log["kin_energy_avg"].append(kin_avg)
            log["threshold_avg"].append(thr_avg)
            diag_str = ""
            if has_diag and cand > 0:
                diag_str = (f"  cand={cand:5d}  ratio={ratio:.3f}  "
                            f"E_kin/W_adh≈{kin_avg / max(thr_avg, 1e-30):.2e}")
            print(
                f"  step {step:5d}/{n_steps}  active={n_a:5d}  attached_cum={n_attached_cum:5d}  "
                f"|ΔM|/M0={err_M:.2e}  events_this_substep={attach.attach_event_count[None]}{diag_str}"
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
        candidates_per_step=np.array(log["candidates_per_step"]),
        commit_ratio=np.array(log["commit_ratio"]),
        kin_energy_avg=np.array(log["kin_energy_avg"]),
        threshold_avg=np.array(log["threshold_avg"]),
        sigma_per_vertex=sigma_snapshot,
        cloth_n_x=cfg.cloth_nx,
        cloth_n_y=cfg.cloth_ny,
    )
    print(f"[OK] log saved to {out_path}")
    print(f"  total attached: {n_attached_cum}, sigma min/max/mean: "
          f"{sigma_snapshot.min():.4e} / {sigma_snapshot.max():.4e} / {sigma_snapshot.mean():.4e}")
    if has_diag and n_candidates_cum > 0:
        overall_ratio = n_attached_cum / n_candidates_cum
        regime = ("JKR-DOMINANT" if overall_ratio < 0.4 else
                  "MIXED" if overall_ratio < 0.85 else
                  "CONTACT-RATE LIMITED ('lucky impact')")
        print(f"  overall commit/candidate ratio: {overall_ratio:.3f}  ⇒  regime: {regime}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
