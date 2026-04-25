"""Plot σ field heatmap + timeseries from `results/attach_demo_log.npz`.

Usage:
    python3 scripts/plot_attach_log.py [--in results/attach_demo_log.npz] [--out results/attach_demo_plots]
"""

import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--in", dest="inp", default="results/attach_demo_log.npz")
    parser.add_argument("--out", default="results/attach_demo_plots")
    args = parser.parse_args()

    log = np.load(args.inp)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1. σ field heatmap (per-vertex, reshaped to cloth grid)
    sigma = log["sigma_per_vertex"]
    n_x = int(log["cloth_n_x"])
    n_y = int(log["cloth_n_y"])
    sigma_grid = sigma.reshape(n_y, n_x)
    fig1, ax1 = plt.subplots(figsize=(6, 5))
    im = ax1.imshow(sigma_grid, origin="lower", cmap="viridis")
    ax1.set_title(f"σ field (kg per vertex), cloth {n_x}×{n_y}")
    ax1.set_xlabel("u (cloth UV)")
    ax1.set_ylabel("v (cloth UV)")
    plt.colorbar(im, ax=ax1, label="σ (kg)")
    fig1.tight_layout()
    fig1.savefig(out_dir / "sigma_heatmap.png", dpi=120)
    plt.close(fig1)

    # 2. Mass drift over time + cumulative attach events
    fig2, (axL, axR) = plt.subplots(1, 2, figsize=(11, 4))
    axL.plot(log["step"], log["mass_drift"], "o-", color="tab:red")
    axL.set_xlabel("step")
    axL.set_ylabel("|M(t)-M_0|/M_0")
    axL.set_title("Mass conservation drift")
    axL.set_yscale("log")
    axL.grid(True, which="both", alpha=0.3)
    axL.axhline(1e-2, color="gray", linestyle="--", label="1% target")
    axL.legend()

    axR.plot(log["step"], log["n_attached_total"], "s-", color="tab:blue", label="cumulative attached")
    axR_twin = axR.twinx()
    axR_twin.plot(log["step"], log["n_active"], "v-", color="tab:orange", label="active sand", alpha=0.7)
    axR.set_xlabel("step")
    axR.set_ylabel("# attached (cumulative)")
    axR_twin.set_ylabel("# active sand")
    axR.set_title("Attach events over time")
    fig2.tight_layout()
    fig2.savefig(out_dir / "timeseries.png", dpi=120)
    plt.close(fig2)

    print(f"[OK] plots saved under {out_dir}")
    print(f"  sigma min/max/mean: {sigma.min():.3e} / {sigma.max():.3e} / {sigma.mean():.3e} (kg)")
    print(f"  final attached: {int(log['n_attached_total'][-1])} of {int(log['n_active'][0]) + int(log['n_attached_total'][-1])} sand particles")
    print(f"  max |ΔM|/M0:    {float(log['mass_drift'].max()):.3e}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
