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

"""
Reproduce all figures from:

    D. Millar et al., "Quench spectroscopy on a digital quantum computer",
    arXiv:2607.02673 (2026). https://arxiv.org/abs/2607.02673

Running this script regenerates the three multi-panel figures from the
experiment data stored in experiments/hardware/.  No arguments are needed.

Figures are saved as PDF and PNG inside the figures/ subdirectory of the
first experiment directory passed to each figure function.

Figure 2  -  FM phase: three single-site quenches (varying Delta) plus one
             two-site quench, framed in a Nature-style 3+1 layout.
Figure 3  -  AFM phase: two single-site quenches at Delta = 2.5 and 5.
Figure 4  -  XY phase: two single-site quenches at Delta = -0.5 and +0.5.
"""

import json
import os

import cmcrameri.cm as cmc
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.patches import FancyBboxPatch
from mpl_toolkits.axes_grid1 import make_axes_locatable

from spectroscopy.fourier import compute_fourier_transform, preprocess_for_fourier
from spectroscopy.hardware_utils import process_observations, read_observation_data
from spectroscopy.plotting import compute_xyz_spectrum

# ---------------------------------------------------------------------------
# Experiment directories (paths relative to repo root)
# ---------------------------------------------------------------------------

BASE = "experiments"

# Figure 2: three single-site (Delta = 3, 2, 1) + one two-site (Delta = 2)
FIG1_DIRS = [
    f"{BASE}/2026_03_10_10_33_45_L_101_Jx_-1.0_Jy_-1.0_Jz_-3.0_hz_0.0__Q_ry_1.571_S50__O_pauli_sy",
    f"{BASE}/2026_03_09_17_00_41_L_101_Jx_-1.0_Jy_-1.0_Jz_-2.0_hz_0.0__Q_ry_1.571_S50__O_pauli_sy",
    f"{BASE}/2026_03_10_10_44_37_L_101_Jx_-1.0_Jy_-1.0_Jz_-1.0_hz_0.0__Q_ry_1.571_S50__O_pauli_sy",
    f"{BASE}/2026_04_07_17_13_09_L_100_Jx_-1.0_Jy_-1.0_Jz_-2.0_hz_0.0__Q_ry_1.571_S49-50__O_flip2_sy",
]

# Figure 2: two single-site AFM quenches (Delta = -2.5, -5)
FIG2_DIRS = [
    f"{BASE}/2026_05_13_17_07_42_L_101_Jx_-1.0_Jy_-1.0_Jz_2.5_hz_0.0__Q_ry_1.571_S50__O_pauli_sy",
    f"{BASE}/2026_05_13_00_13_09_L_101_Jx_-1.0_Jy_-1.0_Jz_5.0_hz_0.0__Q_ry_1.571_S50__O_pauli_sy",
]

# Figure 3: two single-site XY quenches (Delta = 0.5, -0.5)
FIG3_DIRS = [
    f"{BASE}/2026_03_31_01_07_35_L_101_Jx_-1.0_Jy_-1.0_Jz_-0.5_hz_0.0__Q_ry_1.571_S50__O_pauli_sz",
    f"{BASE}/2026_03_30_23_57_48_L_101_Jx_-1.0_Jy_-1.0_Jz_0.5_hz_0.0__Q_ry_1.571_S50__O_pauli_sz",
]

# ---------------------------------------------------------------------------
# Global plot settings
# ---------------------------------------------------------------------------

FONTSIZE = 30
L_THEORY = 1000

mpl.rcParams["figure.figsize"] = (8, 8)
plt.rc("font", family="serif")
plt.rcParams.update(
    {
        "font.size": FONTSIZE,
        "axes.labelsize": FONTSIZE,
        "axes.titlesize": FONTSIZE,
        "xtick.labelsize": FONTSIZE,
        "ytick.labelsize": FONTSIZE,
        "legend.fontsize": FONTSIZE,
    }
)
plt.rc("text", usetex=True)
mpl.rcParams["mathtext.fontset"] = "cm"
mpl.rcParams["mathtext.rm"] = "serif"

# Okabe-Ito colorblind-friendly palette
_OI = {
    "orange": "#E69F00",
    "sky_blue": "#56B4E9",
    "bluish_green": "#009E73",
    "yellow": "#F0E442",
    "blue": "#0072B2",
    "vermillion": "#D55E00",
    "reddish_purple": "#CC79A7",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _results_dir(exp_dir: str) -> str:
    return os.path.join(exp_dir, "results")


def _figures_dir(exp_dir: str) -> str:
    d = os.path.join(exp_dir, "figures")
    os.makedirs(d, exist_ok=True)
    return d


def _format_val(v: float) -> str:
    return f"{int(v)}" if v == int(v) else f"{v:.1f}"


def _baseline_from_info(info: dict) -> float:
    obs_type = info.get("observable_type") or ""
    return 0.5 if obs_type.startswith("projector") else 0.0


def _compose_observable(info: dict) -> str:
    obs_type = info.get("observable_type", "") or ""
    axis = info.get("meas_axis", "") or ""
    proj = info.get("projector") or {}
    if isinstance(proj, dict) and obs_type.startswith("projector"):
        proj_axis = proj.get("axis", axis)
        order, sign = proj.get("order"), proj.get("sign")
        if order:
            return f"{obs_type}:{proj_axis}" if proj_axis else obs_type
        if sign is not None:
            return f"{obs_type}:{proj_axis},sign={sign}" if proj_axis else f"{obs_type},sign={sign}"
        return f"{obs_type}:{proj_axis}" if proj_axis else obs_type
    if obs_type.startswith("flip"):
        return obs_type
    if obs_type == "Dx":
        return "Dx"
    return f"{obs_type}:{axis}" if axis else obs_type


def _zero_white_linear_cmap(cmap, vmin, vmax, name="zwl", n=512):
    """Shift a diverging colormap so its white midpoint falls at data value 0."""
    if vmax <= vmin:
        raise ValueError(f"vmax ({vmax}) must be > vmin ({vmin})")
    if not (vmin < 0.0 < vmax):
        return cmap
    zero_idx = int(np.clip(np.round((0.0 - vmin) / (vmax - vmin) * (n - 1)), 1, n - 2))
    lower = cmap(np.linspace(0.0, 0.5, zero_idx + 1))
    upper = cmap(np.linspace(0.5, 1.0, n - zero_idx))[1:]
    return colors.ListedColormap(np.vstack([lower, upper]), name=name)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def load_experiment(exp_dir: str, *, remove_time_mean: bool) -> dict:
    """Load, symmetrise, and Fourier-transform one experiment directory."""
    res_dir = _results_dir(exp_dir)
    with open(os.path.join(res_dir, "experiment_information.json")) as f:
        info = json.load(f)

    mode = info.get("mode", "hardware")
    L = info["L"]
    slices = info["slices"]
    total_time = info["total_time"]
    Jx, Jy, Jz = info["Jx"], info["Jy"], info["Jz"]
    num_quench_sites = len(info.get("quench", {}).get("sites", [])) or 1

    if mode == "tenpy":
        times = np.load(os.path.join(res_dir, "tenpy_times.npy"))
        obs = np.load(os.path.join(res_dir, "tenpy_observables.npy"))
    else:
        time_resolution = total_time / (slices - 1)
        times = np.linspace(0.0, total_time, slices, endpoint=True)
        anti_file = os.path.join(exp_dir, "results", "obs_antisymmetric.npy")
        if info.get("quench", {}).get("antisymmetric") and os.path.exists(anti_file):
            obs = np.load(anti_file)
        else:
            obs = read_observation_data(exp_dir, time_resolution, slices, L, raw=False)
        obs = process_observations(obs)

    L_eff = obs.shape[1]

    obs_ft = preprocess_for_fourier(
        obs,
        baseline=_baseline_from_info(info),
        remove_time_mean=remove_time_mean,
        remove_spatial_mean=False,
        use_time_derivative=False,
        times=times,
    )
    energies, ft = compute_fourier_transform(obs_ft, total_time, slices, L_eff)

    return {
        "obs": obs,
        "times": times,
        "energies": energies,
        "ft": ft,
        "L_eff": L_eff,
        "Jx": Jx,
        "Jy": Jy,
        "Jz": Jz,
        "observable": _compose_observable(info),
        "num_quench_sites": num_quench_sites,
        "info": info,
    }


# ---------------------------------------------------------------------------
# Dispersion-curve overlay
# ---------------------------------------------------------------------------


def _plot_dispersion(ax, krange, spectra, observable, Jx, Jy, Jz, lw=4.0) -> tuple:
    """Draw Bethe-ansatz curves on *ax*; return (handles, labels) for a legend."""
    is_xxz = np.isclose(Jx, Jy, rtol=1e-10)
    J_perp = Jx if is_xxz else (Jx + Jy) / 2.0
    Delta = Jz / J_perp if J_perp != 0 else np.inf
    is_afm = abs(Delta) >= 1.0
    is_xy = abs(Delta) < 1.0
    is_transverse = any(x in observable.lower() for x in ["sy", "sx", "pauli:sy", "pauli:sx"])

    handles, labels = [], []

    def _add(line, label):
        handles.append(line)
        labels.append(label)

    if is_afm and is_transverse:
        lower = spectra.get("two_spinon_lower")
        upper = spectra.get("two_spinon_upper", lower)
        if lower is not None:
            (ln,) = ax.plot(krange, lower, color="darkorange", ls=":", lw=lw * 1.3, alpha=1.0)
            ax.plot(krange, upper, color="darkorange", ls=":", lw=lw * 1.3, alpha=1.0)
            _add(ln, "Two-spinon continuum bounds")
        bound = spectra.get("two_magnon_bound")
        if isinstance(bound, np.ndarray):
            (ln,) = ax.plot(krange, bound, color="darkorange", ls="--", lw=lw * 0.8, alpha=0.9)
            _add(ln, "Two-spinon bound state")
        magnon = spectra.get("single_magnon")
        if isinstance(magnon, np.ndarray):
            (ln,) = ax.plot(krange, magnon, color="red", ls="--", lw=lw, alpha=0.9)
            _add(ln, "Magnon")

    elif is_xy:
        lower = spectra.get("two_spinon_lower")
        upper = spectra.get("two_spinon_upper")
        if isinstance(lower, np.ndarray) and isinstance(upper, np.ndarray):
            (ln,) = ax.plot(krange, lower, color="darkorange", ls=":", lw=lw * 1.3, alpha=1.0)
            ax.plot(krange, upper, color="darkorange", ls=":", lw=lw * 1.3, alpha=1.0)
            _add(ln, "Two-spinon continuum bounds")
        lsw = spectra.get("single_spinwave_lsw")
        if isinstance(lsw, np.ndarray) and Delta >= 0.0:
            (ln,) = ax.plot(krange, lsw, color=_OI["yellow"], ls="-.", lw=lw, alpha=1.0)
            _add(ln, "Linear spin wave")
        tak = spectra.get("single_spinon_takahashi")
        if isinstance(tak, np.ndarray):
            (ln,) = ax.plot(krange, tak, color="red", ls="--", lw=lw * 1.2, alpha=1.0)
            _add(ln, "Magnon-like")

    else:
        for key, label in [
            ("single_magnon", "Magnon"),
            ("single_spinon_takahashi", "String excitation"),
            ("single_spinwave_lsw", "Single spinon"),
            ("single_spinon", "Single spinon"),
        ]:
            val = spectra.get(key)
            if isinstance(val, np.ndarray):
                (ln,) = ax.plot(krange, val, color="red", ls="--", lw=lw, alpha=0.9)
                _add(ln, label)
                break
        bound_colors = {"two": "darkorange", "three": _OI["reddish_purple"]}
        for key, val in spectra.items():
            if "bound" in key and isinstance(val, np.ndarray):
                pnum = "two" if key.startswith("two") else "three"
                label = (
                    "Two-magnon bound state"
                    if key == "two_magnon_bound"
                    else key.replace("_", " ").title()
                )
                (ln,) = ax.plot(
                    krange, val, color=bound_colors.get(pnum, "purple"), ls="--", lw=lw, alpha=0.9
                )
                _add(ln, label)

    return handles, labels


# ---------------------------------------------------------------------------
# Column renderer (shared by all three figures)
# ---------------------------------------------------------------------------


def _render_column(
    ax_dyn,
    ax_qsf,
    data: dict,
    vmin_dyn,
    vmax_dyn,
    vmin_qsf,
    vmax_qsf,
    *,
    is_leftmost: bool,
    add_colorbar: bool,
    draw_single_magnon: bool = True,
    qsf_ylim: tuple | None = (-3.0, 3.0),
) -> tuple:
    """
    Fill one column of a multi-panel figure (dynamics on top, QSF below).
    Returns (legend_handles, legend_labels).
    """
    obs = data["obs"]
    times = data["times"]
    energies = data["energies"]
    ft = data["ft"]
    L_eff = data["L_eff"]
    Jx, Jy, Jz = data["Jx"], data["Jy"], data["Jz"]
    observable = data["observable"]
    nqs = data["num_quench_sites"]

    dyn_norm = colors.Normalize(vmin=vmin_dyn, vmax=vmax_dyn)
    dyn_cmap = _zero_white_linear_cmap(cmc.vik, vmin_dyn, vmax_dyn)  # ty: ignore[unresolved-attribute]

    # -- Dynamics --
    x = np.arange(L_eff)
    X, T = np.meshgrid(x, times)
    im1 = ax_dyn.pcolormesh(
        X, T, obs, shading="auto", cmap=dyn_cmap, norm=dyn_norm, rasterized=True
    )
    ax_dyn.set_xlabel(r"$r$")
    if is_leftmost:
        ax_dyn.set_ylabel(r"$tJ$")
    else:
        ax_dyn.tick_params(labelleft=False)
    Delta = -Jz / abs(Jx)
    ax_dyn.set_title(rf"$\Delta = {_format_val(Delta)}$", fontsize=FONTSIZE, pad=8)

    if add_colorbar:
        div1 = make_axes_locatable(ax_dyn)
        cax1 = div1.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im1, cax=cax1).set_label(r"$G(r, t)$")

    # -- QSF --
    krange = np.linspace(-np.pi, np.pi, L_eff, endpoint=False)
    dk = krange[1] - krange[0]
    energy_scale = energies / abs(Jx)
    im2 = ax_qsf.pcolormesh(
        krange + dk / 2,
        energy_scale,
        ft[::-1],
        shading="auto",
        cmap=cmc.oslo_r,  # ty: ignore[unresolved-attribute]
        vmin=vmin_qsf,
        vmax=vmax_qsf,
        rasterized=True,
    )
    ax_qsf.set_xlabel(r"$k$")
    if is_leftmost:
        ax_qsf.set_ylabel(r"$\omega/J$")
    else:
        ax_qsf.tick_params(labelleft=False)
    ax_qsf.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax_qsf.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    if qsf_ylim is not None:
        ax_qsf.set_ylim(qsf_ylim)

    krange_th, spectra = compute_xyz_spectrum(
        L=L_eff,
        Jx=Jx,
        Jy=Jy,
        Jz=Jz,
        hz=0.0,
        staggered=False,
        num_quench_sites=nqs,
        L_theory=L_THEORY,
    )
    # Suppress magnon overlay for two-site observable if requested
    if not draw_single_magnon:
        spectra.pop("single_magnon", None)

    handles, labels = _plot_dispersion(ax_qsf, krange_th, spectra, observable, Jx, Jy, Jz)

    if add_colorbar:
        div2 = make_axes_locatable(ax_qsf)
        cax2 = div2.append_axes("right", size="5%", pad=0.1)
        plt.colorbar(im2, cax=cax2).set_label(r"$G(k, \omega)$")

    return handles, labels


# ---------------------------------------------------------------------------
# Figure 1  –  3 single-site + 1 two-site, framed Nature layout
# ---------------------------------------------------------------------------


def _draw_frame(fig, axes_list, label, pad=(0.009, 0.055, 0.030, 0.018), lw=1.4, ec="1.0"):
    """Draw a rounded rectangle around the union tight bbox of axes_list."""
    renderer = fig.canvas.get_renderer()
    bboxes = [ax.get_tightbbox(renderer) for ax in axes_list]
    x0 = min(b.x0 for b in bboxes)
    y0 = min(b.y0 for b in bboxes)
    x1 = max(b.x1 for b in bboxes)
    y1 = max(b.y1 for b in bboxes)

    # Absorb adjacent colorbars into the frame
    ax_centers = [0.5 * (ax.get_position().x0 + ax.get_position().x1) for ax in axes_list]
    own = set(axes_list)
    for ax in fig.axes:
        if ax in own:
            continue
        bb = ax.get_tightbbox(renderer)
        horiz = 0 <= bb.x0 - x1 <= 80 or 0 <= x0 - bb.x1 <= 80
        vert = not (bb.y1 < y0 - 5 or bb.y0 > y1 + 5)
        if not (horiz and vert):
            continue
        cx = 0.5 * (ax.get_position().x0 + ax.get_position().x1)
        others = [a for a in fig.axes if a not in own and a is not ax]
        nearest_own = min(abs(cx - c) for c in ax_centers)
        nearest_other = min(
            (abs(cx - 0.5 * (a.get_position().x0 + a.get_position().x1)) for a in others),
            default=float("inf"),
        )
        if nearest_own <= nearest_other:
            x0, y0, x1, y1 = min(x0, bb.x0), min(y0, bb.y0), max(x1, bb.x1), max(y1, bb.y1)

    inv = fig.transFigure.inverted()
    fx0, fy0 = inv.transform((x0, y0))
    fx1, fy1 = inv.transform((x1, y1))
    pl, pr, pt, pb = pad
    fig.add_artist(
        FancyBboxPatch(
            (fx0 - pl, fy0 - pb),
            (fx1 - fx0) + pl + pr,
            (fy1 - fy0) + pt + pb,
            transform=fig.transFigure,
            boxstyle="round,pad=0,rounding_size=0.012",
            linewidth=lw,
            edgecolor=ec,
            facecolor="none",
            zorder=0,
            clip_on=False,
        )
    )
    fig.text(
        fx0 - pl + 0.008,
        fy0 - pb + (fy1 - fy0) + pt + pb - 0.008,
        label,
        fontsize=FONTSIZE + 2,
        fontweight="bold",
        fontfamily="Helvetica",
        ha="left",
        va="top",
        usetex=False,
    )


def plot_figure1(dirs: list[str], output_path: str, *, legend_anchor_y: float = -0.06) -> None:
    """3 single-site columns + 1 two-site column, framed."""
    print("\n-- figures/fig2 --")
    single_data = [load_experiment(d, remove_time_mean=False) for d in dirs[:3]]
    two_data = load_experiment(dirs[3], remove_time_mean=False)

    vmin_s = min(d["obs"].min() for d in single_data)
    vmax_s = max(d["obs"].max() for d in single_data)
    vmin_s_qsf = min(d["ft"].min() for d in single_data)
    vmax_s_qsf = max(d["ft"].max() for d in single_data)
    vmin_t, vmax_t = two_data["obs"].min(), two_data["obs"].max()
    vmin_t_qsf, vmax_t_qsf = two_data["ft"].min(), two_data["ft"].max()

    fig = plt.figure(figsize=(28, 11.5))
    main_gs = fig.add_gridspec(
        2,
        2,
        hspace=0.32,
        wspace=0.14,
        width_ratios=[3, 1],
        left=0.06,
        right=0.975,
        top=0.90,
        bottom=0.10,
    )
    gs_a = main_gs[:, 0].subgridspec(2, 3, hspace=0.32, wspace=0.07)
    gs_b = main_gs[:, 1].subgridspec(2, 1, hspace=0.32)

    a_axes_dyn, a_axes_qsf = [], []
    a_legend: dict[str, Artist] = {}
    b_legend: dict[str, Artist] = {}
    ax_dyn0: Axes | None = None
    ax_qsf0: Axes | None = None

    for idx, data in enumerate(single_data):
        ax_dyn = fig.add_subplot(
            gs_a[0, idx], **({"sharex": ax_dyn0, "sharey": ax_dyn0} if idx else {})
        )
        ax_qsf = fig.add_subplot(
            gs_a[1, idx], **({"sharex": ax_qsf0, "sharey": ax_qsf0} if idx else {})
        )
        if idx == 0:
            ax_dyn0, ax_qsf0 = ax_dyn, ax_qsf
        h, l = _render_column(
            ax_dyn,
            ax_qsf,
            data,
            vmin_s,
            vmax_s,
            vmin_s_qsf,
            vmax_s_qsf,
            is_leftmost=(idx == 0),
            add_colorbar=(idx == 2),
            draw_single_magnon=True,
            qsf_ylim=(-5.0, 5.0),
        )
        for hh, ll in zip(h, l):
            a_legend.setdefault(ll, hh)
        a_axes_dyn.append(ax_dyn)
        a_axes_qsf.append(ax_qsf)

    ax_dyn_b = fig.add_subplot(gs_b[0, 0])
    ax_qsf_b = fig.add_subplot(gs_b[1, 0])
    h, l = _render_column(
        ax_dyn_b,
        ax_qsf_b,
        two_data,
        vmin_t,
        vmax_t,
        vmin_t_qsf,
        vmax_t_qsf,
        is_leftmost=False,
        add_colorbar=True,
        draw_single_magnon=False,
        qsf_ylim=(-5.0, 5.0),
    )
    for hh, ll in zip(h, l):
        b_legend.setdefault(ll, hh)

    # Sync QSF y-axis: clamp data range to [-5, 5] (matches reference script)
    energy_min = min(d["energies"].min() / abs(d["Jx"]) for d in single_data)
    energy_max = max(d["energies"].max() / abs(d["Jx"]) for d in single_data)
    energy_min = max(energy_min, -5.0)
    energy_max = min(energy_max, 5.0)
    assert ax_qsf0 is not None
    ax_qsf0.set_ylim(energy_min, energy_max)
    ax_qsf_b.set_ylim(energy_min, energy_max)

    fig.canvas.draw()
    _draw_frame(fig, a_axes_dyn + a_axes_qsf, "a")
    _draw_frame(fig, [ax_dyn_b, ax_qsf_b], "b")

    # Separate legends per panel
    if "Magnon" in a_legend:
        fig.legend(
            [a_legend["Magnon"]],
            ["Magnon"],
            loc="lower center",
            bbox_to_anchor=(0.375, legend_anchor_y),
            ncol=1,
            frameon=True,
            framealpha=0.95,
            edgecolor="0.3",
            fontsize=FONTSIZE,
            handlelength=2.0,
        )
    bound_items = {l: h for l, h in b_legend.items() if "bound" in l.lower()}
    if bound_items:
        fig.legend(
            list(bound_items.values()),
            list(bound_items.keys()),
            loc="lower center",
            bbox_to_anchor=(0.875, legend_anchor_y),
            ncol=1,
            frameon=True,
            framealpha=0.95,
            edgecolor="0.3",
            fontsize=FONTSIZE,
            handlelength=2.0,
        )

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Figures 2 & 3  –  generic N-column layout
# ---------------------------------------------------------------------------


def plot_figure_multi(
    dirs: list[str],
    output_path: str,
    *,
    remove_time_mean: bool,
    draw_single_magnon: bool = True,
    qsf_ylim: tuple | None = (-3.0, 3.0),
    legend_anchor_y: float = 0.0,
) -> None:
    """Two (or more) single-site columns with a shared bottom legend."""
    n = len(dirs)
    print(f"\n-- {output_path} --")
    all_data = [load_experiment(d, remove_time_mean=remove_time_mean) for d in dirs]

    vmin_dyn = min(d["obs"].min() for d in all_data)
    vmax_dyn = min(max(d["obs"].max() for d in all_data), 0.5)
    vmin_qsf = min(d["ft"].min() for d in all_data)
    vmax_qsf = max(d["ft"].max() for d in all_data)

    fig = plt.figure(figsize=(7 * n, 10))
    gs = fig.add_gridspec(2, n, hspace=0.3, wspace=0.15, bottom=0.15)

    ax_dyn0: Axes | None = None
    ax_qsf0: Axes | None = None
    legend_entries: dict[str, Artist] = {}

    for idx, data in enumerate(all_data):
        kw_dyn = {"sharex": ax_dyn0, "sharey": ax_dyn0} if idx else {}
        kw_qsf = {"sharex": ax_qsf0, "sharey": ax_qsf0} if idx else {}
        ax_dyn = fig.add_subplot(gs[0, idx], **kw_dyn)
        ax_qsf = fig.add_subplot(gs[1, idx], **kw_qsf)
        if idx == 0:
            ax_dyn0, ax_qsf0 = ax_dyn, ax_qsf

        h, l = _render_column(
            ax_dyn,
            ax_qsf,
            data,
            vmin_dyn,
            vmax_dyn,
            vmin_qsf,
            vmax_qsf,
            is_leftmost=(idx == 0),
            add_colorbar=(idx == n - 1),
            draw_single_magnon=draw_single_magnon,
            qsf_ylim=qsf_ylim,
        )
        for hh, ll in zip(h, l):
            legend_entries.setdefault(ll, hh)

    if legend_entries:
        ncol = max(1, (len(legend_entries) + 1) // 2)
        fig.legend(
            list(legend_entries.values()),
            list(legend_entries.keys()),
            loc="lower center",
            bbox_to_anchor=(0.5, legend_anchor_y),
            ncol=ncol,
            frameon=True,
            framealpha=0.95,
            edgecolor="0.3",
            fontsize=FONTSIZE,
            handlelength=2.0,
        )

    _save(fig, output_path)


# ---------------------------------------------------------------------------
# Save helper
# ---------------------------------------------------------------------------


def _save(fig, path: str) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    for ext, kw in [(".pdf", {})]:
        p = path + ext
        fig.savefig(p, bbox_inches="tight", **kw)
        print(f"  Saved {p}")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    plot_figure1(FIG1_DIRS, output_path=os.path.join("figures", "fig2"), legend_anchor_y=-0.07)
    plot_figure_multi(
        FIG2_DIRS,
        output_path=os.path.join("figures", "fig3"),
        remove_time_mean=True,
        draw_single_magnon=False,
        qsf_ylim=None,
        legend_anchor_y=-0.05,
    )
    plot_figure_multi(
        FIG3_DIRS,
        output_path=os.path.join("figures", "fig4"),
        remove_time_mean=True,
        draw_single_magnon=False,
        legend_anchor_y=-0.11,
    )
