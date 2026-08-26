# (C) Copyright IBM 2026.
# (C) Copyright UKRI-STFC (Hartree Centre) 2026.
#
# This code is licensed under the Apache License, Version 2.0. You may
# obtain a copy of this license in the LICENSE.txt file in the root directory
# of this source tree or at http://www.apache.org/licenses/LICENSE-2.0.
#
# Any modifications or derivative works of this code must retain this
# copyright notice, and modified files need to carry a notice indicating
# that they have been altered from the originals.

import numpy as np
from scipy import optimize, special


# Helper function for wrapping momentum to [-π, π)
def wrap_to_pi(x):
    """Helper function for wrapping momentum to [-π, π)"""
    return (x + np.pi) % (2 * np.pi) - np.pi


def _two_spinon_term_afm(
    q: np.ndarray,
    J_perp: float,
    Delta: float,
) -> np.ndarray:
    """
    Per-spinon energy contribution as it appears in Takahashi (4.70) for the
    gapped XXZ chain (|Delta| > 1).

    Reference:
        M. Takahashi, "Thermodynamics of One-Dimensional Solvable Models,"
        Cambridge University Press (1999), Eq. (4.70).

    Each spinon contributes
        epsilon(q) = (K(u) sinh(gamma) / pi) * |J_perp| * sqrt(1 - u^2 cos^2(q)),
    where Delta = cosh(gamma) and the modulus u solves K'(u)/K(u) = gamma/pi.
    """
    if Delta <= 1.0:
        raise ValueError("AFM two-spinon term requires Delta > 1")

    gamma = np.arccosh(Delta)
    target = float(gamma / np.pi)

    def _ratio(k: float) -> float:
        kp_sq = max(0.0, 1.0 - k * k)
        return float(special.ellipk(kp_sq) / special.ellipk(k * k))

    u = float(optimize.brentq(lambda k: _ratio(k) - target, 1e-12, 1.0 - 1e-12, maxiter=200))
    K_ellip = float(special.ellipk(u * u))

    pref = (K_ellip * np.sinh(gamma) / np.pi) * abs(J_perp)
    return pref * np.sqrt(1.0 - (u**2) * np.cos(q) ** 2)


def compute_xyz_spectrum(
    L: int,
    Jx: float,
    Jy: float,
    Jz: float,
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    staggered: bool = False,
    num_quench_sites: int = 1,
    L_theory: int | None = None,
) -> tuple[np.ndarray, dict[str, np.ndarray | None]]:
    """
    Compute Bethe-ansatz-inspired spectra for XXZ/XYZ spin chains.
    """
    krange = np.linspace(-np.pi, np.pi, L, endpoint=False)
    spectra: dict[str, np.ndarray | None] = {}

    is_xxz = np.isclose(Jx, Jy, rtol=1e-10)

    has_field = not (np.isclose(hx, 0.0) and np.isclose(hy, 0.0) and np.isclose(hz, 0.0))

    if is_xxz:
        J_perp = Jx
        Delta = Jz / J_perp if J_perp != 0 else np.inf
    else:
        J_perp = (Jx + Jy) / 2.0
        Delta = Jz / J_perp if J_perp != 0 else np.inf

    J_scale = abs(J_perp)

    print("J_scale", J_scale, "Delta", Delta)

    # Phase classification based on Delta and sign of J_perp
    # For positive J_perp (FM XY coupling): Delta > 1 → AFM, Delta < -1 → FM
    # For negative J_perp (AFM XY coupling): Delta > 1 → FM, Delta < -1 → AFM
    if J_perp > 0:
        # Positive J_perp (ferromagnetic XY plane)
        if Delta >= 1.0:
            phase = "AFM"
            particle_name = "magnon"
        elif Delta <= -1.0:
            phase = "FM"
            particle_name = "magnon"
            Delta = abs(Delta)
        elif abs(Delta) < 1.0:
            phase = "XY"
            particle_name = "spinon"
        else:
            phase = "critical"
            particle_name = "spinon"
    else:
        # Negative J_perp (antiferromagnetic XY plane)
        if Delta <= -1.0:
            phase = "AFM"
            particle_name = "magnon"
            Delta = abs(Delta)
        elif Delta >= 1.0:
            phase = "FM"
            particle_name = "magnon"
        elif abs(Delta) < 1.0:
            phase = "XY"
            particle_name = "spinon"
        else:
            phase = "critical"
            particle_name = "spinon"

    print(
        "Jx,Jz",
        Jx,
        Jz,
        "J_perp",
        J_perp,
        "J_scale",
        J_scale,
        "Delta",
        Delta,
        "phase",
        phase,
    )

    # ========== SINGLE-PARTICLE EXCITATIONS ==========
    if phase == "AFM":
        # AFM phase: elementary excitations are spinons. Store the per-spinon
        # des Cloizeaux-Pearson dispersion as single_magnon for use as a
        # reference line on the QSF.
        particle_name = "spinon"  # Override particle name for AFM
        single_magnon = _two_spinon_term_afm(krange, J_perp=J_perp, Delta=abs(Delta))
        spectra["single_magnon"] = single_magnon

    elif phase == "XY":
        J_eff = J_scale

        # Eq. (12): linear spin-wave dispersion about the x-polarized initial state
        S = 0.5
        spectrum_lsw = (
            2.0 * J_eff * S * np.sqrt((1.0 - np.cos(krange)) * (1.0 - Delta * np.cos(krange)))
        )

        if has_field and not np.isclose(hz, 0.0):
            spectrum_lsw = np.sqrt(spectrum_lsw**2 + hz**2)

        spectra["single_spinwave_lsw"] = spectrum_lsw

        # gamma is shared between the Takahashi curve and the continuum boundaries.
        gamma = np.arccos(np.clip(-Delta, -1.0, 1.0))

        # Only include the dressed single-spinon guide when it will be drawn (Jz <= 0).
        if Jz <= 0.0:
            # Eq. (13): dressed single-spinon dispersion (guide curve only)
            # In our convention, the Takahashi anisotropy is effectively -Delta.
            p0 = np.pi / gamma
            angle = (p0 - 1.0) * np.pi / 2.0
            tan_angle = np.tan(angle)
            if np.isclose(tan_angle, 0.0, atol=1e-12):
                spectrum_takahashi = None
            else:
                cot_term = 1.0 / tan_angle
                spectrum_takahashi = (
                    (np.pi * J_eff * np.sin(gamma) / gamma)
                    * np.abs(np.sin(krange / 2.0))
                    * np.sqrt(1.0 + (cot_term**2) * np.sin(krange / 2.0) ** 2)
                )
            if has_field and not np.isclose(hz, 0.0) and spectrum_takahashi is not None:
                spectrum_takahashi = np.sqrt(spectrum_takahashi**2 + hz**2)
            spectra["single_spinon_takahashi"] = spectrum_takahashi

        # Analytic XY two-spinon continuum boundaries
        # These should always be used in the XY phase.
        two_lower = (np.pi * J_eff * np.sin(gamma) / (2.0 * gamma)) * np.abs(np.sin(krange))
        two_upper = (np.pi * J_eff * np.sin(gamma) / gamma) * np.abs(np.sin(krange / 2.0))

        if has_field and not np.isclose(hz, 0.0):
            two_lower = np.sqrt(two_lower**2 + hz**2)
            two_upper = np.sqrt(two_upper**2 + hz**2)

        spectra["two_spinon_lower"] = two_lower
        spectra["two_spinon_upper"] = two_upper
        spectra["two_spinon_lower_boundary"] = two_lower
        spectra["two_spinon_upper_boundary"] = two_upper

    elif phase == "FM":
        spectrum = -Jx * (1 - np.cos(krange)) - (1 + Jz)
        if has_field and not np.isclose(hz, 0.0):
            spectrum = spectrum + abs(hz)
        spectra["single_magnon"] = spectrum

    else:  # critical
        spectrum = J_scale * np.abs(np.sin(krange))
        spectra["single_spinon"] = spectrum

    # ========== CONTINUUM KINEMATICS ==========
    if phase == "XY":
        # Already handled analytically above
        return krange, spectra

    # ========== TWO-PARTICLE EXCITATIONS ==========
    if num_quench_sites >= 2 or phase == "AFM":
        # For AFM, we compute two-spinon continuum directly using elliptic dispersion
        # For other phases, we need the single particle spectrum first
        single_key = f"single_{particle_name}"

        if phase != "AFM":
            if single_key not in spectra or spectra[single_key] is None:
                return krange, spectra
            single_val = spectra[single_key]
            assert isinstance(single_val, np.ndarray)
            single = single_val
        L_calc = L_theory if L_theory is not None else max(400, L * 4)
        krange_calc = np.linspace(-np.pi, np.pi, L_calc, endpoint=False)

        if phase == "AFM":
            # Per-spinon term from Takahashi (4.70)
            elem_calc = _two_spinon_term_afm(krange_calc, J_perp=J_perp, Delta=abs(Delta))
        else:
            elem_calc = np.interp(krange_calc, krange, single, period=2 * np.pi)

        two_lower_calc = np.zeros(L_calc)
        two_upper_calc = np.zeros(L_calc)

        def elem_at(q):
            q = wrap_to_pi(q)
            idx = np.argmin(np.abs(krange_calc - q))
            return elem_calc[idx]

        for i, K in enumerate(krange_calc):
            omega_vals = np.zeros(L_calc)
            for j, q in enumerate(krange_calc):
                # AFM Takahashi shift:
                q2 = K + np.pi - q
                omega_vals[j] = elem_at(q) + elem_at(q2)

            two_lower_calc[i] = np.min(omega_vals)
            # two_upper_calc[i] = np.max(omega_vals)
            two_upper_calc[i] = 2 * elem_at((K + np.pi) / 2)

        two_lower_calc = np.where(np.isinf(two_lower_calc), 0.0, two_lower_calc)
        two_upper_calc = np.where(np.isinf(two_upper_calc), 0.0, two_upper_calc)

        two_lower = np.interp(krange, krange_calc, two_lower_calc, period=2 * np.pi)
        two_upper = np.interp(krange, krange_calc, two_upper_calc, period=2 * np.pi)

        spectra["two_spinon_lower"] = two_lower
        spectra["two_spinon_upper"] = two_upper
        spectra[f"two_{particle_name}_lower"] = two_lower
        spectra[f"two_{particle_name}_upper"] = two_upper

        if particle_name == "magnon" and abs(Delta) > 1.0:
            eta = np.arccosh(abs(Delta))
            sinh_eta = np.sinh(eta)
            sinh_2eta = np.sinh(2 * eta)
            cosh_2eta = np.cosh(2 * eta)

            if abs(sinh_2eta) > 1e-10:
                bound_2 = J_scale * (sinh_eta / sinh_2eta) * (cosh_2eta - np.cos(krange))
                spectra["two_magnon_bound"] = bound_2
            else:
                spectra["two_magnon_bound"] = None

    # ========== THREE-PARTICLE EXCITATIONS ==========
    if num_quench_sites >= 3:
        k_third_idx = np.array([np.argmin(np.abs(krange - k / 3)) for k in krange])
        spectra[f"three_{particle_name}_lower"] = 3 * single[k_third_idx]

        E_min_three = np.min(single)
        k_min_idx = np.argmin(single)
        k_min_val = krange[k_min_idx]

        k_remaining_idx = np.array(
            [np.argmin(np.abs(krange - (k - 2 * k_min_val))) for k in krange]
        )
        spectra[f"three_{particle_name}_upper"] = 2 * E_min_three + single[k_remaining_idx]

        if f"two_{particle_name}_upper" in spectra:
            two_upper_val = spectra[f"two_{particle_name}_upper"]
            assert isinstance(two_upper_val, np.ndarray)
            k_shifted_idx = np.array([np.argmin(np.abs(krange - (k - k_min_val))) for k in krange])
            alt_upper = E_min_three + two_upper_val[k_shifted_idx]
            three_upper_val = spectra[f"three_{particle_name}_upper"]
            assert isinstance(three_upper_val, np.ndarray)
            spectra[f"three_{particle_name}_upper"] = np.maximum(three_upper_val, alt_upper)

        if phase == "AFM" and abs(Delta) > 1.0:
            nu = np.arccos(1.0 / abs(Delta))
            if abs(np.sin(3 * nu)) > 1e-10:
                bound_3 = (
                    J_scale * (3 * np.sin(nu) / np.sin(3 * nu)) * (np.cos(3 * nu) - np.cos(krange))
                )
                spectra["three_magnon_bound"] = bound_3
            else:
                spectra["three_magnon_bound"] = None

    # ========== STAGGERED FIELD EFFECTS ==========
    if staggered and has_field:
        gap = np.sqrt(hx**2 + hy**2 + hz**2)
        for key in list(spectra.keys()):
            val = spectra[key]
            if isinstance(val, np.ndarray):
                gap_profile = gap * np.abs(np.sin(krange))
                spectra[key] = np.sqrt(val**2 + gap_profile**2)

    return krange, spectra
