"""JKR (Johnson-Kendall-Roberts) adhesion closed-form helpers.

Reference: Johnson, Kendall & Roberts 1971 (Proc. R. Soc. A);
Maugis 1992 (J. Colloid Interface Sci.) for regularization.

Two parallel implementations:
- numpy reference (for testing and Jupyter exploration)
- Taichi @ti.func (for kernel use in Operator A/B)

Both must agree to ULP — see tests/test_jkr_formula.py.
"""

import math

import numpy as np
import taichi as ti

_PI = math.pi


def jkr_pulloff_work_np(radius: float, gamma: float, k_reduced: float) -> float:
    """JKR pull-off work in joules (numpy reference).

    W = 2γ
    a_c = (9π R² W / 8K)^(1/3)
    E_sep = (2/5)π W a_c²

    Args:
        radius:    grain radius R (m)
        gamma:     surface energy γ (J/m²)
        k_reduced: reduced modulus K (Pa) — combines grain & cloth Young's moduli.

    Returns:
        Pull-off work (J).
    """
    if radius <= 0 or gamma <= 0 or k_reduced <= 0:
        return 0.0
    contact_radius = (
        9.0 * _PI * gamma * radius * radius / (4.0 * k_reduced)
    ) ** (1.0 / 3.0)
    return (4.0 * _PI * gamma / 5.0) * contact_radius * contact_radius


def jkr_pulloff_work_array(
    radius: np.ndarray, gamma: np.ndarray, k_reduced: np.ndarray
) -> np.ndarray:
    """Vectorized numpy version for parameter sweeps."""
    radius, gamma, k_reduced = np.broadcast_arrays(radius, gamma, k_reduced)
    result = np.zeros_like(radius, dtype=np.result_type(radius, gamma, k_reduced, float))
    mask = (radius > 0) & (gamma > 0) & (k_reduced > 0)
    contact_radius = np.cbrt(
        9.0 * _PI * gamma[mask] * radius[mask] * radius[mask] / (4.0 * k_reduced[mask])
    )
    result[mask] = (4.0 * _PI * gamma[mask] / 5.0) * contact_radius * contact_radius
    return result


def gamma_humid_np(
    gamma_0: float,
    beta: float,
    humidity: float,
    sigma_local: float,
    sigma_max: float,
) -> float:
    """Capillary saturation: γ depends on humidity h and local saturation σ/σ_max.

    γ_T(h, σ) = γ_0 + β · h · (1 - σ_local / σ_max)

    The (1 - σ/σ_max) factor saturates: a saturated patch can't gain more capillary
    boost. Humidity h ∈ [0, 1].
    """
    saturation_factor = max(0.0, 1.0 - sigma_local / max(sigma_max, 1e-12))
    return gamma_0 + beta * max(0.0, min(1.0, humidity)) * saturation_factor


# ---- Taichi versions (callable from kernels) -----------------------------------------

@ti.func
def jkr_pulloff_work(radius, gamma, k_reduced):
    """JKR pull-off work — Taichi @ti.func form (untyped, inferred at call site)."""
    result = 0.0
    if radius > 0.0 and gamma > 0.0 and k_reduced > 0.0:
        contact_radius = ti.pow(
            9.0 * _PI * gamma * radius * radius / (4.0 * k_reduced),
            1.0 / 3.0,
        )
        result = (4.0 * _PI * gamma / 5.0) * contact_radius * contact_radius
    return result


@ti.func
def gamma_humid(gamma_0, beta, humidity, sigma_local, sigma_max):
    """Humidity-dependent surface energy — Taichi @ti.func form."""
    saturation_factor = ti.max(0.0, 1.0 - sigma_local / ti.max(sigma_max, 1e-12))
    h_clamped = ti.max(0.0, ti.min(1.0, humidity))
    return gamma_0 + beta * h_clamped * saturation_factor
