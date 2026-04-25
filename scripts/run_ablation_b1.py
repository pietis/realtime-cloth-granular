"""B1 ablation — side-by-side comparison of JKR Operator A vs Heuristic-stick baseline.

Runs both operators on the same scene configuration + seed, collects σ fields,
prints summary metrics, saves comparison plot + npz log.

Note: Both operators run in the same Taichi session (ti.init called once).
Each run allocates fresh particle/cloth fields; between runs we reinitialize
the same pre-allocated buffers so memory usage stays bounded.

Run:
    python scripts/run_ablation_b1.py --steps 20 --cpu
    python scripts/run_ablation_b1.py --steps 200

Outputs:
    results/ablation_b1/comparison.png   -- side-by-side sigma heatmaps
    results/ablation_b1/b1_log.npz       -- logged arrays for downstream analysis
"""

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def run_one(
    operator_name,
    particles, cloth, sand, contact, attach,
    cfg, n_steps,
):
    """Run n_steps of simulation with the given pre-built operator set.

    Returns (summary_dict, sigma_per_vertex_array).
    Reads initial mass/momentum before the loop; both fields are already
    initialized by the caller.
    """
    from src.mpm.particles import count_active
    from src.utils.conservation import relative_error, total_mass, total_linear_momentum
    from src.utils.visualize import per_vertex_sigma_numpy

    M0 = float(total_mass(particles, cloth.triangles))
    P0 = total_linear_momentum(particles, cloth.vertices)
    P0_norm = float(P0.norm())

    n_attached_cum = 0
    sub_dt = cfg.dt / cfg.n_substeps

    for _step in range(n_steps):
        attach.reset_audit()
        for _ in range(cfg.n_substeps):
            sand.step(sub_dt)
            cloth.step(sub_dt)
            contact.solve()
            attach.step()
        n_attached_cum += int(attach.attach_event_count[None])

    M_final = float(total_mass(particles, cloth.triangles))
    P_final = total_linear_momentum(particles, cloth.vertices)
    mass_drift = relative_error(M_final, M0)
    P_final_norm = float(P_final.norm())
    mom_drift = abs(P_final_norm - P0_norm) / max(abs(P0_norm), 1.0)

    sigma_arr = per_vertex_sigma_numpy(cloth)
    sigma_total = float(sigma_arr.sum())

    # Shannon entropy of normalised sigma distribution (measure of spatial spread)
    sigma_flat = sigma_arr.flatten()
    sigma_pos = sigma_flat[sigma_flat > 0]
    if len(sigma_pos) > 1:
        p = sigma_pos / sigma_pos.sum()
        sigma_entropy = float(-np.sum(p * np.log(p + 1.0e-12)))
    else:
        sigma_entropy = 0.0

    summary = {
        "operator": operator_name,
        "total_attached": n_attached_cum,
        "mass_drift": mass_drift,
        "mom_drift": mom_drift,
        "sigma_total": sigma_total,
        "sigma_entropy": sigma_entropy,
        "sigma_min": float(sigma_arr.min()),
        "sigma_max": float(sigma_arr.max()),
        "sigma_mean": float(sigma_arr.mean()),
    }
    return summary, sigma_arr


def build_scene(cfg, operator_name):
    """Allocate all Taichi fields and solvers for one operator run.

    Both runs share the same ti.init() but allocate independent Taichi
    StructFields so they don't interfere.
    """
    import taichi as ti

    from src.cloth.xpbd import ClothSolver
    from src.coupling.attach import AttachOperator
    from src.coupling.attach_heuristic import HeuristicAttachOperator
    from src.coupling.contact import ContactSolver
    from src.mpm.grid import Grid
    from src.mpm.particles import init_particles_box, make_particle_field
    from src.mpm.sand import SandSolver

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
        n_x=cfg.cloth_nx,
        n_y=cfg.cloth_ny,
        size_x=cfg.cloth_size,
        size_y=cfg.cloth_size,
        origin=cfg.cloth_origin,
        density=cfg.cloth_density,
    )
    if cfg.pinned_corners and any(cfg.pinned_corners):
        bl, br, tl, tr = cfg.pinned_corners
        n_x, n_y = cfg.cloth_nx, cfg.cloth_ny
        pin = []
        if bl:
            pin.append(0)
        if br:
            pin.append(n_x - 1)
        if tl:
            pin.append((n_y - 1) * n_x)
        if tr:
            pin.append((n_y - 1) * n_x + (n_x - 1))
        if pin:
            cloth.pin_vertices(pin)

    contact = ContactSolver(particles, cloth, contact_radius=cfg.jkr.contact_radius)

    if operator_name == "heuristic":
        attach = HeuristicAttachOperator(
            particles, cloth,
            gamma_0=cfg.jkr.gamma_0,
            beta=cfg.jkr.beta,
            humidity=cfg.jkr.humidity,
            sigma_max=cfg.jkr.sigma_max,
            k_reduced=cfg.jkr.k_reduced,
            lam=cfg.jkr.lam,
            contact_radius=cfg.jkr.contact_radius,
        )
    else:
        attach = AttachOperator(
            particles, cloth,
            gamma_0=cfg.jkr.gamma_0,
            beta=cfg.jkr.beta,
            humidity=cfg.jkr.humidity,
            sigma_max=cfg.jkr.sigma_max,
            k_reduced=cfg.jkr.k_reduced,
            lam=cfg.jkr.lam,
            contact_radius=cfg.jkr.contact_radius,
        )

    return particles, cloth, sand, contact, attach


def main():
    parser = argparse.ArgumentParser(
        description="B1 ablation: JKR Operator A vs Heuristic-stick side-by-side"
    )
    parser.add_argument("--config", type=str, default="data/configs/demo_a_lying.yaml")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="results/ablation_b1")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load config (pure Python — no Taichi yet)
    from src.utils.config import load_scene_config
    cfg = load_scene_config(args.config)

    # Initialize Taichi once for the whole process
    import taichi as ti
    if args.cpu:
        ti.init(arch=ti.cpu)
    else:
        try:
            ti.init(arch=ti.cuda)
        except Exception:
            ti.init(arch=ti.cpu)

    print(
        f"[b1_ablation] config={args.config}  steps={args.steps}  seed={args.seed}"
    )

    # --- Run 1: JKR ---
    print("[b1_ablation] Building JKR Operator A scene ...")
    particles_jkr, cloth_jkr, sand_jkr, contact_jkr, attach_jkr = build_scene(cfg, "jkr")
    print("[b1_ablation] Running JKR Operator A ...")
    summary_jkr, sigma_jkr = run_one(
        "jkr", particles_jkr, cloth_jkr, sand_jkr, contact_jkr, attach_jkr,
        cfg, args.steps,
    )

    # --- Run 2: Heuristic ---
    print("[b1_ablation] Building B1 Heuristic-stick scene ...")
    particles_h, cloth_h, sand_h, contact_h, attach_h = build_scene(cfg, "heuristic")
    print("[b1_ablation] Running B1 Heuristic-stick baseline ...")
    summary_h, sigma_h = run_one(
        "heuristic", particles_h, cloth_h, sand_h, contact_h, attach_h,
        cfg, args.steps,
    )

    # --- Print summary table ---
    print("\n===== B1 Ablation Summary =====")
    col_w = 22
    metrics = [
        ("total_attached", "total_attached"),
        ("mass_drift",     "mass_drift"),
        ("mom_drift",      "mom_drift"),
        ("sigma_total",    "sigma_total"),
        ("sigma_entropy",  "sigma_entropy"),
        ("sigma_max",      "sigma_max"),
    ]
    print(f"  {'metric':<{col_w}}  {'JKR (Operator A)':<{col_w}}  {'Heuristic (B1)':<{col_w}}")
    print("  " + "-" * (col_w * 3 + 4))
    for label, key in metrics:
        jv = summary_jkr[key]
        hv = summary_h[key]
        if isinstance(jv, float):
            jv = f"{jv:.4e}"
            hv = f"{hv:.4e}"
        print(f"  {label:<{col_w}}  {str(jv):<{col_w}}  {str(hv):<{col_w}}")
    print("================================\n")

    # --- Save npz ---
    npz_path = out_dir / "b1_log.npz"
    np.savez(
        npz_path,
        sigma_jkr=sigma_jkr,
        sigma_heuristic=sigma_h,
        cloth_nx=cfg.cloth_nx,
        cloth_ny=cfg.cloth_ny,
        steps=args.steps,
        seed=args.seed,
        jkr_total_attached=summary_jkr["total_attached"],
        jkr_mass_drift=summary_jkr["mass_drift"],
        jkr_sigma_total=summary_jkr["sigma_total"],
        jkr_sigma_entropy=summary_jkr["sigma_entropy"],
        heuristic_total_attached=summary_h["total_attached"],
        heuristic_mass_drift=summary_h["mass_drift"],
        heuristic_sigma_total=summary_h["sigma_total"],
        heuristic_sigma_entropy=summary_h["sigma_entropy"],
    )
    print(f"[OK] log saved to {npz_path}")

    # --- Plot ---
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        nx, ny = cfg.cloth_nx, cfg.cloth_ny
        sigma_jkr_2d = sigma_jkr.reshape(ny, nx)
        sigma_h_2d = sigma_h.reshape(ny, nx)
        vmax = max(float(sigma_jkr_2d.max()), float(sigma_h_2d.max()), 1.0e-9)

        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        fig.suptitle(
            f"B1 Ablation: sigma heatmap after {args.steps} steps  (seed={args.seed})",
            fontsize=13,
        )

        im0 = axes[0].imshow(sigma_jkr_2d, origin="lower", vmin=0, vmax=vmax, cmap="hot")
        axes[0].set_title(
            f"JKR Operator A\n"
            f"attached={summary_jkr['total_attached']}  "
            f"sigma_total={summary_jkr['sigma_total']:.3e}\n"
            f"mass_drift={summary_jkr['mass_drift']:.2e}  "
            f"entropy={summary_jkr['sigma_entropy']:.3f}"
        )
        axes[0].set_xlabel("cloth X")
        axes[0].set_ylabel("cloth Y")
        plt.colorbar(im0, ax=axes[0], label="sigma (kg) per vertex")

        im1 = axes[1].imshow(sigma_h_2d, origin="lower", vmin=0, vmax=vmax, cmap="hot")
        axes[1].set_title(
            f"B1 Heuristic-stick\n"
            f"attached={summary_h['total_attached']}  "
            f"sigma_total={summary_h['sigma_total']:.3e}\n"
            f"mass_drift={summary_h['mass_drift']:.2e}  "
            f"entropy={summary_h['sigma_entropy']:.3f}"
        )
        axes[1].set_xlabel("cloth X")
        axes[1].set_ylabel("cloth Y")
        plt.colorbar(im1, ax=axes[1], label="sigma (kg) per vertex")

        plt.tight_layout()
        plot_path = out_dir / "comparison.png"
        plt.savefig(plot_path, dpi=120)
        plt.close(fig)
        print(f"[OK] comparison plot saved to {plot_path}")
    except Exception as exc:
        print(f"[WARN] matplotlib plot failed: {exc}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
