"""JKR pull-off work — numpy reference vs Taichi @ti.func cross-check.

Both must agree to within ULP-level tolerance over a parameter sweep.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def test_jkr_closed_form_basic():
    """Reference numpy implementation against a hand-computed value."""
    from src.coupling.jkr import jkr_pulloff_work_np

    R = 1.0e-3       # 1 mm grain
    gamma = 0.05     # 50 mJ/m²
    K = 1.0e6        # 1 MPa reduced modulus
    w = jkr_pulloff_work_np(R, gamma, K)

    # Hand check: prefactor = 15/(8π) ≈ 0.5968
    # factor    = (3π · 0.05 · 1e-3 / 4e6)^(1/3) ≈ (1.178e-10)^(1/3) ≈ 4.91e-4
    # core      = π · R² · 2γ ≈ π · 1e-6 · 0.1 ≈ 3.14e-7
    # total ≈ 0.5968 · 3.14e-7 · 4.91e-4 ≈ 9.20e-11
    expected = (15.0 / (8.0 * math.pi)) * math.pi * R * R * 2.0 * gamma * \
               (3.0 * math.pi * gamma * R / (4.0 * K)) ** (1.0 / 3.0)
    assert math.isclose(w, expected, rel_tol=1e-12)
    assert 1e-12 < w < 1e-9


def test_jkr_taichi_matches_numpy():
    """Cross-check: Taichi kernel form must match numpy reference within float32."""
    import taichi as ti

    from src.coupling.jkr import jkr_pulloff_work, jkr_pulloff_work_np

    ti.init(arch=ti.cpu)

    test_field = ti.field(ti.f32, shape=())

    @ti.kernel
    def call_taichi(R: ti.f32, gamma: ti.f32, K: ti.f32):
        test_field[None] = jkr_pulloff_work(R, gamma, K)

    R_vals     = [5e-4, 1e-3, 2e-3, 5e-3]
    gamma_vals = [0.02, 0.05, 0.10, 0.20]
    K_vals     = [5e5, 1e6, 5e6]

    for R in R_vals:
        for g in gamma_vals:
            for K in K_vals:
                ref = jkr_pulloff_work_np(R, g, K)
                call_taichi(R, g, K)
                got = float(test_field[None])
                # float32 precision tolerance
                assert math.isclose(got, ref, rel_tol=1e-5, abs_tol=1e-15), (
                    f"mismatch at R={R}, γ={g}, K={K}: ref={ref}, got={got}"
                )


def test_humidity_increases_gamma():
    """γ_humid(h>0) > γ_0 when σ < σ_max; equals γ_0 when h=0 or saturated."""
    from src.coupling.jkr import gamma_humid_np

    g0, beta, h, sig, sig_max = 0.05, 0.10, 0.5, 0.0, 0.1
    g_dry  = gamma_humid_np(g0, beta, 0.0, sig, sig_max)
    g_humid = gamma_humid_np(g0, beta, h, sig, sig_max)
    g_saturated = gamma_humid_np(g0, beta, h, sig_max, sig_max)

    assert math.isclose(g_dry, g0, rel_tol=1e-12)
    assert g_humid > g0
    assert math.isclose(g_saturated, g0, rel_tol=1e-12)


def test_jkr_monotonic_in_radius():
    """W_adh ∝ R^(7/3) ⇒ strictly monotone in R when γ, K fixed."""
    from src.coupling.jkr import jkr_pulloff_work_array

    R = np.array([5e-4, 1e-3, 2e-3, 5e-3, 1e-2])
    gamma = np.full_like(R, 0.05)
    K = np.full_like(R, 1e6)
    w = jkr_pulloff_work_array(R, gamma, K)
    assert np.all(np.diff(w) > 0), f"non-monotonic: {w}"
