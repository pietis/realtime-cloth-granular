"""γ (surface energy) sweep — cleanest single-knob test of the JKR threshold.

Holds R, geometry, mass, λ, and σ_max fixed; varies γ_0 only. Theory
predicts attach fraction ∝ γ^(4/3) before σ_max saturation kicks in
(W_adh ∝ R^(7/3) γ^(4/3) K^(-1/3)).

This sweep is *easier* to fit than R-sweep because:
  - No geometric coupling: contact_radius and gap aren't R-coupled.
  - Single physical knob.
  - Direct test of the closed-form jkr_pulloff_work().

Run:
    python3 scripts/sweep_gamma.py --steps 150 --cpu \\
        --config jkr_dominant.yaml \\
        --gammas "0.5,1.0,2.0,4.0,8.0" --seeds "42,43,44" \\
        --lam 1.0 --out results/gamma_sweep
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_one_gamma(gamma_0: float, steps: int, cpu: bool,
                  config_name: str = "jkr_dominant.yaml",
                  seed: int = 42,
                  lam_override: float | None = None,
                  humidity: float = 0.0) -> dict:
    """Run a single demo at the given γ_0, return attach stats."""
    if cpu:
        ti.init(arch=ti.cpu, random_seed=seed)
    else:
        try:
            ti.init(arch=ti.cuda, random_seed=seed)
        except Exception:
            ti.init(arch=ti.cpu, random_seed=seed)

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
    from src.utils.conservation import total_mass
    from src.utils.visualize import per_vertex_sigma_numpy

    repo = Path(__file__).resolve().parent.parent
    cfg = load_scene_config(repo / "data" / "configs" / config_name)
    cfg.jkr.gamma_0 = gamma_0
    cfg.jkr.humidity = humidity
    if lam_override is not None:
        cfg.jkr.lam = lam_override

    particles = make_particle_field(cfg.max_particles)
    init_particles_box(
        particles, cfg.n_active_particles,
        ti.Vector(cfg.sand_box_min), ti.Vector(cfg.sand_box_max),
        cfg.mass_per_particle, cfg.particle_radius,
    )
    grid = Grid(cfg.grid_res, cfg.domain_size)
    sand = SandSolver(particles, grid)

    cloth = ClothSolver(
        n_x=cfg.cloth_nx, n_y=cfg.cloth_ny,
        size_x=cfg.cloth_size, size_y=cfg.cloth_size,
        origin=cfg.cloth_origin, density=cfg.cloth_density,
    )
    bl, br, tl, tr = cfg.pinned_corners
    n_x, n_y = cfg.cloth_nx, cfg.cloth_ny
    pin = []
    if bl: pin.append(0)
    if br: pin.append(n_x - 1)
    if tl: pin.append((n_y - 1) * n_x)
    if tr: pin.append((n_y - 1) * n_x + (n_x - 1))
    if pin: cloth.pin_vertices(pin)

    contact = ContactSolver(particles, cloth, contact_radius=cfg.jkr.contact_radius)
    attach_op = AttachOperator(
        particles, cloth,
        gamma_0=cfg.jkr.gamma_0, beta=cfg.jkr.beta,
        humidity=cfg.jkr.humidity, sigma_max=cfg.jkr.sigma_max,
        k_reduced=cfg.jkr.k_reduced, lam=cfg.jkr.lam,
        contact_radius=cfg.jkr.contact_radius,
    )

    M0 = float(total_mass(particles, cloth.triangles))
    sub_dt = cfg.dt / cfg.n_substeps
    n_attached_cum = 0
    n_candidates_cum = 0
    has_diag = hasattr(attach_op, "candidate_count")
    for step in range(steps):
        attach_op.reset_audit()
        for _ in range(cfg.n_substeps):
            sand.step(sub_dt)
            cloth.step(sub_dt)
            attach_op.step()        # JKR before contact
            contact.solve()
        n_attached_cum += int(attach_op.attach_event_count[None])
        if has_diag:
            n_candidates_cum += int(attach_op.candidate_count[None])

    M_final = float(total_mass(particles, cloth.triangles))
    sigma = per_vertex_sigma_numpy(cloth)
    commit_ratio = (n_attached_cum / max(n_candidates_cum, 1)) if has_diag else float("nan")

    return {
        "gamma_0": gamma_0,
        "n_attached": int(n_attached_cum),
        "n_candidates": int(n_candidates_cum),
        "commit_ratio": float(commit_ratio),
        "n_active_initial": int(cfg.n_active_particles),
        "attach_fraction": n_attached_cum / max(cfg.n_active_particles, 1),
        "mass_drift": abs(M_final - M0) / M0 if M0 > 0 else 0.0,
        "sigma_total": float(sigma.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=150)
    parser.add_argument("--gammas", type=str, default="0.5,1.0,2.0,4.0,8.0",
                        help="Surface energies (J/m²) — log-spaced for clean R^(4/3) fit")
    parser.add_argument("--humidity", type=float, default=0.0,
                        help="Held fixed; γ_0 is the swept knob.")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--config", type=str, default="jkr_dominant.yaml")
    parser.add_argument("--seeds", type=str, default="42,43,44")
    parser.add_argument("--lam", type=float, default=None,
                        help="Override λ. Set ~1.0 to disable distance penalty.")
    parser.add_argument("--out", default="results/gamma_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    g_list = [float(x) for x in args.gammas.split(",")]
    seed_list = [int(x) for x in args.seeds.split(",")]

    grid = []
    for g in g_list:
        per_seed = []
        for s in seed_list:
            print(f"\n==== γ = {g:.2f} J/m², seed = {s} ====")
            row = run_one_gamma(g, args.steps, args.cpu, args.config, s,
                                args.lam, args.humidity)
            print(f"  attached: {row['n_attached']}, fraction: {row['attach_fraction']:.4f}, "
                  f"ratio: {row['commit_ratio']:.3f}, ΔM/M₀: {row['mass_drift']:.2e}")
            per_seed.append(row)
        grid.append(per_seed)

    G = np.array(g_list)
    seed_arr = np.array(seed_list)
    fraction_grid = np.array([[r["attach_fraction"] for r in row] for row in grid])
    fraction_mean = fraction_grid.mean(axis=1)
    fraction_std = fraction_grid.std(axis=1)
    commit_grid = np.array([[r["commit_ratio"] for r in row] for row in grid])
    commit_mean = commit_grid.mean(axis=1)
    mass_drift = np.array([[r["mass_drift"] for r in row] for row in grid]).mean(axis=1)

    np.savez(out_dir / "gamma_sweep.npz",
             gamma=G, seeds=seed_arr,
             fraction_grid=fraction_grid,
             fraction=fraction_mean, fraction_std=fraction_std,
             commit_ratio_grid=commit_grid, commit_ratio_mean=commit_mean,
             mass_drift=mass_drift,
             humidity=args.humidity, steps=args.steps)

    # Theory: attach fraction ∝ γ^(4/3) (in JKR-dominant, non-saturated regime)
    valid = fraction_mean > 0
    slope_str = "n/a"
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(np.log(G[valid]), np.log(fraction_mean[valid]), 1)
        ss_res = np.sum((np.log(fraction_mean[valid]) - (slope * np.log(G[valid]) + intercept))**2)
        ss_tot = np.sum((np.log(fraction_mean[valid]) - np.log(fraction_mean[valid]).mean())**2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        slope_str = f"slope={slope:.3f} (theory γ^(4/3) ≈ 1.33), R²={r2:.3f}"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(12, 4.5))
    if len(seed_list) > 1:
        axL.errorbar(G, fraction_mean, yerr=fraction_std, fmt="o-",
                     color="tab:blue", linewidth=2, capsize=4,
                     label=f"mean ± std ({len(seed_list)} seeds)")
        for s_idx in range(len(seed_list)):
            axL.scatter(G, fraction_grid[:, s_idx], alpha=0.4, s=15, color="tab:blue")
    else:
        axL.plot(G, fraction_mean, "o-", color="tab:blue", linewidth=2, label="observed")
    axL.set_xscale("log")
    axL.set_yscale("log")
    if valid.sum() >= 2:
        G_fit = np.geomspace(G.min(), G.max(), 100)
        axL.plot(G_fit, np.exp(intercept) * G_fit ** slope, "--", color="gray",
                 label=f"fit (slope={slope:.2f})")
        axL.plot(G_fit, fraction_mean[valid][0] * (G_fit / G[valid][0]) ** (4/3),
                 ":", color="tab:red", label="theory γ^(4/3)")
    axL.set_xlabel("γ_0 (J/m²)")
    axL.set_ylabel("attach fraction")
    axL.set_title(f"Attach fraction vs γ_0 (h={args.humidity})\n{slope_str}")
    axL.grid(True, which="both", alpha=0.3)
    axL.legend()

    axR.plot(G, commit_mean, "s-", color="tab:purple", linewidth=2)
    axR.set_xlabel("γ_0 (J/m²)")
    axR.set_ylabel("commit/candidate ratio")
    axR.set_xscale("log")
    axR.set_title("Commit ratio vs γ_0\n(rises ⇒ JKR threshold easier to pass)")
    axR.grid(True, which="both", alpha=0.3)
    axR.axhspan(0.4, 0.85, alpha=0.1, color="green", label="MIXED regime")
    axR.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "gamma_sweep.png", dpi=120)
    plt.close(fig)

    print(f"\n[OK] saved to {out_dir}/gamma_sweep.{{npz,png}}")
    print(f"  log-log fit: {slope_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
