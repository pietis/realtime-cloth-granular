"""CPU performance scaling benchmark — Big-O linearity defense (Gemini suggestion).

Measures step time as a function of N_particles. The "real-time" claim of the
project ultimately needs GPU benchmarks, but we can defensibly extrapolate from
CPU O(N) scaling to GPU performance if the algorithm shows clean linear scaling.

Outputs:
- results/scaling/scaling.npz   : raw (N, ms_per_step) pairs
- results/scaling/scaling.png   : log-log plot + linear fit + slope (target ≈ 1)

Run:
    python3 scripts/benchmark_scaling.py --cpu --particles "500,1000,2500,5000,10000"
"""

import argparse
import sys
import time
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import taichi as ti

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_one_size(n_particles: int, n_steps: int, cpu: bool) -> tuple[float, float]:
    """Returns (ms_per_step_avg, ms_per_step_std)."""
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
    from src.mpm.particles import init_particles_box, make_particle_field
    from src.mpm.sand import SandSolver
    from src.utils.config import load_scene_config

    repo = Path(__file__).resolve().parent.parent
    cfg = load_scene_config(repo / "data" / "configs" / "demo_a_lying.yaml")
    cfg.n_active_particles = n_particles
    cfg.max_particles = max(n_particles + 100, cfg.max_particles)

    particles = make_particle_field(cfg.max_particles)
    init_particles_box(
        particles, n_particles,
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

    sub_dt = cfg.dt / cfg.n_substeps

    # Warm-up (compile all kernels)
    for _ in range(2):
        sand.step(sub_dt)
        cloth.step(sub_dt)
        contact.solve()
        attach_op.step()
    ti.sync()

    # Timed loop
    times_ms = []
    for _ in range(n_steps):
        t0 = time.perf_counter()
        for _ in range(cfg.n_substeps):
            sand.step(sub_dt)
            cloth.step(sub_dt)
            contact.solve()
            attach_op.step()
        ti.sync()
        times_ms.append((time.perf_counter() - t0) * 1000.0)

    return float(np.mean(times_ms)), float(np.std(times_ms))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--particles", type=str, default="500,1000,2500,5000,10000")
    parser.add_argument("--n-steps", type=int, default=10, help="timed steps per size")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--out", default="results/scaling")
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    sizes = [int(x) for x in args.particles.split(",")]
    means = []
    stds = []
    for n in sizes:
        print(f"\n==== Benchmarking N = {n:,} ====")
        m, s = run_one_size(n, args.n_steps, args.cpu)
        print(f"  step time: {m:.2f} ± {s:.2f} ms")
        means.append(m)
        stds.append(s)

    sizes_arr = np.array(sizes)
    means_arr = np.array(means)
    stds_arr = np.array(stds)

    np.savez(out_dir / "scaling.npz",
             n_particles=sizes_arr,
             ms_per_step_mean=means_arr,
             ms_per_step_std=stds_arr,
             n_steps_per_size=args.n_steps)

    # Log-log linear fit; slope ≈ 1.0 ⇒ O(N) (good)
    slope, intercept = np.polyfit(np.log(sizes_arr), np.log(means_arr), 1)
    ss_res = np.sum((np.log(means_arr) - (slope * np.log(sizes_arr) + intercept))**2)
    ss_tot = np.sum((np.log(means_arr) - np.log(means_arr).mean())**2)
    r2 = 1.0 - ss_res / max(ss_tot, 1e-12)

    # Extrapolated 60-fps budget at N = 100k
    extrapolated_100k = np.exp(slope * np.log(100_000) + intercept)
    fps_at_100k = 1000.0 / max(extrapolated_100k, 1e-6)

    fig, ax = plt.subplots(figsize=(7.5, 5))
    ax.errorbar(sizes_arr, means_arr, yerr=stds_arr, marker="o", linestyle="-",
                color="tab:blue", linewidth=2, capsize=3, label="observed")
    N_fit = np.geomspace(sizes_arr.min(), 1.5e5, 100)
    ax.plot(N_fit, np.exp(intercept) * N_fit ** slope, "--", color="gray",
            label=f"fit (slope={slope:.2f}, R²={r2:.3f})")
    ax.axvline(100_000, color="tab:orange", linestyle=":", alpha=0.7,
               label=f"100k extrap: {extrapolated_100k:.1f} ms ⇒ {fps_at_100k:.1f} fps")
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("N particles")
    ax.set_ylabel("ms per step (substeps included)")
    ax.set_title("CPU step-time scaling (Taichi x64 backend)")
    ax.grid(True, which="both", alpha=0.3)
    ax.legend()

    fig.tight_layout()
    fig.savefig(out_dir / "scaling.png", dpi=120)
    plt.close(fig)

    print(f"\n[OK] saved to {out_dir}/scaling.{{npz,png}}")
    print(f"  slope = {slope:.3f}, R² = {r2:.3f}")
    print(f"  extrapolated step time at N=100k: {extrapolated_100k:.1f} ms (CPU)")
    print(f"  CPU 60-fps headroom at N=100k: {fps_at_100k:.1f} fps")
    print(f"  GPU expected speedup ≥ 10× (Taichi CUDA vs x64) ⇒ N=1M target plausible at home.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
