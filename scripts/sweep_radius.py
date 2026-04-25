"""Step 6 (partial) — Codex S1: grain-radius sweep over JKR attach demo.

Theory: W_adh ∝ R^(7/3). Holding humidity, impact velocity, and scene
constant, attach count should rise strongly with R. This is the cleanest
sanity check for the JKR formula being applied correctly downstream.

Output: npz log + plot (attach count vs R, log-log).

Run:
    python3 scripts/sweep_radius.py --steps 60 --cpu --radii "0.0025,0.0035,0.005,0.0075,0.01"
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_one_radius(radius: float, steps: int, cpu: bool, humidity: float = 0.6,
                   config_name: str = "demo_a_lying.yaml") -> dict:
    if cpu:
        ti.init(arch=ti.cpu)
    else:
        try:
            ti.init(arch=ti.cuda)
        except Exception:
            ti.init(arch=ti.cpu)

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
    cfg.particle_radius = radius
    cfg.jkr.humidity = humidity
    # IMPORTANT: hold contact_radius and λ fixed across the sweep so the
    # only thing that varies is R itself. If contact_radius scales with R,
    # larger particles have more far-distance candidates that decay via
    # exp(-d/λ) and the slope of attach-fraction vs R gets flattened.
    # (Codex feedback after first sweep showed slope=0.27 instead of theoretical 2.33.)

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
            attach_op.step()        # JKR before contact (read pre-bounce velocity)
            contact.solve()
        n_attached_cum += int(attach_op.attach_event_count[None])
        if has_diag:
            n_candidates_cum += int(attach_op.candidate_count[None])

    M_final = float(total_mass(particles, cloth.triangles))
    sigma = per_vertex_sigma_numpy(cloth)
    commit_ratio = (n_attached_cum / max(n_candidates_cum, 1)) if has_diag else float("nan")

    return {
        "radius": radius,
        "n_attached": int(n_attached_cum),
        "n_candidates": int(n_candidates_cum),
        "commit_ratio": float(commit_ratio),
        "n_active_initial": int(cfg.n_active_particles),
        "attach_fraction": n_attached_cum / max(cfg.n_active_particles, 1),
        "mass_drift": abs(M_final - M0) / M0 if M0 > 0 else 0.0,
        "sigma_total": float(sigma.sum()),
        "sigma_max": float(sigma.max()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument("--radii", type=str, default="0.0025,0.0035,0.005,0.0075,0.01",
                        help="Comma-separated grain radii in meters")
    parser.add_argument("--humidity", type=float, default=0.6)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--config", type=str, default="demo_a_lying.yaml",
                        help="Config name in data/configs/ (e.g. jkr_dominant.yaml)")
    parser.add_argument("--out", default="results/radius_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    r_list = [float(x) for x in args.radii.split(",")]

    rows = []
    for r in r_list:
        print(f"\n==== Running R = {r*1e3:.2f} mm ====")
        row = run_one_radius(r, args.steps, args.cpu, args.humidity, args.config)
        print(f"  attached: {row['n_attached']}, fraction: {row['attach_fraction']:.4f}, "
              f"σ_total: {row['sigma_total']:.4e}, ΔM/M₀: {row['mass_drift']:.2e}, "
              f"commit_ratio: {row['commit_ratio']:.3f}")
        rows.append(row)

    R = np.array([r["radius"] for r in rows])
    n_attached = np.array([r["n_attached"] for r in rows])
    fraction = np.array([r["attach_fraction"] for r in rows])
    mass_drift = np.array([r["mass_drift"] for r in rows])

    np.savez(out_dir / "radius_sweep.npz",
             radius=R, n_attached=n_attached, fraction=fraction,
             mass_drift=mass_drift, humidity=args.humidity, steps=args.steps)

    # Theoretical: log(N) = (7/3) log(R) + const if dominated by JKR threshold scaling.
    # Fit slope of log(fraction) vs log(R) (ignore zeros).
    valid = fraction > 0
    fit_str = "n/a"
    if valid.sum() >= 2:
        slope, intercept = np.polyfit(np.log(R[valid]), np.log(fraction[valid]), 1)
        ss_res = np.sum((np.log(fraction[valid]) - (slope * np.log(R[valid]) + intercept))**2)
        ss_tot = np.sum((np.log(fraction[valid]) - np.log(fraction[valid]).mean())**2)
        r2 = 1.0 - ss_res / max(ss_tot, 1e-12)
        fit_str = f"slope={slope:.3f} (theory 7/3 ≈ 2.33), R²={r2:.3f}"

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    axL.loglog(R * 1e3, fraction, "o-", color="tab:blue", linewidth=2, label="observed")
    if valid.sum() >= 2:
        R_fit = np.linspace(R.min(), R.max(), 100)
        axL.loglog(R_fit * 1e3, np.exp(intercept) * R_fit ** slope, "--", color="gray",
                   label=f"fit (slope={slope:.2f})")
        # Reference theoretical slope
        axL.loglog(R_fit * 1e3, (R_fit / R[valid][0]) ** (7/3) * fraction[valid][0],
                   ":", color="tab:red", label="theory R^(7/3)")
    axL.set_xlabel("grain radius R (mm)")
    axL.set_ylabel("attach fraction")
    axL.set_title(f"Attach fraction vs radius (h={args.humidity})\n{fit_str}")
    axL.grid(True, which="both", alpha=0.3)
    axL.legend()

    axR.plot(R * 1e3, mass_drift, "s-", color="tab:red", linewidth=2)
    axR.set_xlabel("grain radius R (mm)")
    axR.set_ylabel("|ΔM|/M₀")
    axR.set_yscale("log")
    axR.set_title("Mass conservation across radius sweep")
    axR.axhline(1e-2, color="gray", linestyle="--", label="1% target")
    axR.grid(True, which="both", alpha=0.3)
    axR.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "radius_sweep.png", dpi=120)
    plt.close(fig)

    print(f"\n[OK] saved to {out_dir}/radius_sweep.{{npz,png}}")
    print(f"  log-log fit: {fit_str}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
