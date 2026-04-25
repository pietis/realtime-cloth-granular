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


def run_one_humidity(humidity: float, steps: int, cpu: bool) -> dict:
    """Run a single attach demo at the given humidity and return summary stats."""
    # Taichi must be re-init'd between runs to clear allocated fields.
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
    cfg = load_scene_config(repo / "data" / "configs" / "demo_a_lying.yaml")
    cfg.jkr.humidity = humidity   # override sweep variable

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
    for step in range(steps):
        attach_op.reset_audit()
        for _ in range(cfg.n_substeps):
            sand.step(sub_dt)
            cloth.step(sub_dt)
            contact.solve()
            attach_op.step()
        n_attached_cum += int(attach_op.attach_event_count[None])

    M_final = float(total_mass(particles, cloth.triangles))
    sigma = per_vertex_sigma_numpy(cloth)

    return {
        "humidity": humidity,
        "n_attached": int(n_attached_cum),
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
    parser.add_argument("--out", default="results/humidity_sweep")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    h_list = [float(x) for x in args.humidities.split(",")]

    rows = []
    for h in h_list:
        print(f"\n==== Running humidity = {h} ====")
        row = run_one_humidity(h, args.steps, args.cpu)
        print(f"  attached: {row['n_attached']}, sigma_total: {row['sigma_total']:.4e}, mass_drift: {row['mass_drift']:.2e}")
        rows.append(row)

    h_arr = np.array([r["humidity"] for r in rows])
    n_attached = np.array([r["n_attached"] for r in rows])
    sigma_total = np.array([r["sigma_total"] for r in rows])
    mass_drift = np.array([r["mass_drift"] for r in rows])

    np.savez(
        out_dir / "humidity_sweep.npz",
        humidity=h_arr,
        n_attached=n_attached,
        sigma_total=sigma_total,
        mass_drift=mass_drift,
        steps=args.steps,
    )

    fig, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4.5))
    axL.plot(h_arr, n_attached, "o-", color="tab:blue", linewidth=2)
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

    # Step 6 gate metric: monotonicity check
    is_monotone = bool(np.all(np.diff(n_attached) >= 0))
    print(f"  monotone attach-count(humidity): {is_monotone}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
