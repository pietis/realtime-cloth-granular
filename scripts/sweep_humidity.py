"""Step 6 (partial) — humidity sweep over JKR attach demo.

Runs the same scene at h ∈ {0.0, 0.25, 0.5, 0.75, 1.0} and records final
attach count + σ statistics. This is the minimum quantitative gate for
"capillary saturation parameter actually controls attach density" (계획서.md §3
Q1, demo (b)).

Output: a single npz log + 1 plot showing attach count vs humidity.

Run:
    python3 scripts/sweep_humidity.py --steps 50 [--cpu]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_one_humidity(humidity: float, steps: int, cpu: bool,
                     config_name: str = "demo_a_lying.yaml",
                     seed: int = 42, lam_override: float | None = None) -> dict:
    """Run a single attach demo at the given humidity and return summary stats."""
    # Taichi must be re-init'd between runs to clear allocated fields.
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
    cfg.jkr.humidity = humidity   # override sweep variable
    if lam_override is not None:
        cfg.jkr.lam = lam_override

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
    if cfg.pinned_corners and any(cfg.pinned_corners):
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
            attach_op.step()
            contact.solve()
        n_attached_cum += int(attach_op.attach_event_count[None])
        if has_diag:
            n_candidates_cum += int(attach_op.candidate_count[None])

    M_final = float(total_mass(particles, cloth.triangles))
    sigma = per_vertex_sigma_numpy(cloth)

    commit_ratio = (n_attached_cum / max(n_candidates_cum, 1)) if has_diag else float("nan")
    return {
        "humidity": humidity,
        "n_attached": int(n_attached_cum),
        "n_candidates": int(n_candidates_cum),
        "commit_ratio": float(commit_ratio),
        "n_active_final": int(count_active(particles)),
        "n_active_initial": int(cfg.n_active_particles),
        "mass_drift": abs(M_final - M0) / M0 if M0 > 0 else 0.0,
        "sigma_max":  float(sigma.max()),
        "sigma_mean": float(sigma.mean()),
        "sigma_total": float(sigma.sum()),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--steps", type=int, default=40)
    parser.add_argument("--humidities", type=str, default="0.0,0.25,0.5,0.75,1.0")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--config", type=str, default="demo_a_lying.yaml")
    parser.add_argument("--seeds", type=str, default="42",
                        help="Comma-separated seeds. Multiple seeds give error bars.")
    parser.add_argument("--lam", type=float, default=None,
                        help="Override JKR distance falloff λ (m). Set ~1.0 to disable.")
    parser.add_argument("--out", default="results/humidity_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    h_list = [float(x) for x in args.humidities.split(",")]
    seed_list = [int(x) for x in args.seeds.split(",")]

    # Run all (humidity × seed) combinations.
    # Layout: rows[h_idx][s_idx] = single-run dict
    grid = []
    for h in h_list:
        per_seed = []
        for s in seed_list:
            print(f"\n==== Running humidity = {h}, seed = {s} ====")
            row = run_one_humidity(h, args.steps, args.cpu, args.config, s, args.lam)
            print(f"  attached: {row['n_attached']} (cand={row['n_candidates']}, ratio={row['commit_ratio']:.3f}), "
                  f"σ_total: {row['sigma_total']:.4e}, ΔM/M₀: {row['mass_drift']:.2e}")
            per_seed.append(row)
        grid.append(per_seed)

    h_arr = np.array(h_list)
    seed_arr = np.array(seed_list)
    n_attached_grid = np.array([[r["n_attached"] for r in row_seeds] for row_seeds in grid])
    n_attached_mean = n_attached_grid.mean(axis=1)
    n_attached_std  = n_attached_grid.std(axis=1)
    sigma_total = np.array([[r["sigma_total"] for r in row_seeds] for row_seeds in grid]).mean(axis=1)
    mass_drift  = np.array([[r["mass_drift"]  for r in row_seeds] for row_seeds in grid]).mean(axis=1)
    commit_ratio_grid = np.array([[r["commit_ratio"] for r in row_seeds] for row_seeds in grid])

    np.savez(
        out_dir / "humidity_sweep.npz",
        humidity=h_arr,
        seeds=seed_arr,
        n_attached_grid=n_attached_grid,
        n_attached=n_attached_mean,           # backwards-compat name
        n_attached_mean=n_attached_mean,
        n_attached_std=n_attached_std,
        sigma_total=sigma_total,
        mass_drift=mass_drift,
        commit_ratio_grid=commit_ratio_grid,
        steps=args.steps,
    )

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    if len(seed_list) > 1:
        axL.errorbar(h_arr, n_attached_mean, yerr=n_attached_std, fmt="o-",
                     color="tab:blue", linewidth=2, capsize=4,
                     label=f"mean ± std over {len(seed_list)} seeds")
        # also show individual seeds as faint dots
        for s_idx, s in enumerate(seed_list):
            axL.scatter(h_arr, n_attached_grid[:, s_idx], alpha=0.4, s=15,
                        color="tab:blue")
        axL.legend()
    else:
        axL.plot(h_arr, n_attached_mean, "o-", color="tab:blue", linewidth=2)
    axL.set_xlabel("humidity h")
    axL.set_ylabel("# attached after sweep")
    axL.set_title("Attach count vs humidity (capillary saturation effect)")
    axL.grid(True, alpha=0.3)

    axR.plot(h_arr, mass_drift, "s-", color="tab:red", linewidth=2)
    axR.set_xlabel("humidity h")
    axR.set_ylabel("|ΔM|/M₀")
    axR.set_yscale("log")
    axR.set_title("Mass conservation across humidity")
    axR.axhline(1e-2, color="gray", linestyle="--", label="1% target")
    axR.grid(True, which="both", alpha=0.3)
    axR.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "humidity_sweep.png", dpi=120)
    plt.close(fig)

    print(f"\n[OK] saved to {out_dir}/humidity_sweep.{{npz,png}}")

    # Step 6 gate metric: monotonicity check on the mean
    is_monotone = bool(np.all(np.diff(n_attached_mean) >= 0))
    sn = (n_attached_mean.max() - n_attached_mean.min()) / max(n_attached_std.mean(), 1e-9)
    print(f"  monotone attach-count(humidity): {is_monotone}")
    print(f"  signal/noise (range/avg-std): {sn:.2f}  (>2 means signal exceeds noise)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
