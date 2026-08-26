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

"""Minimal post-run plot script.

Usage:
    python scripts/plot_results.py <experiment_directory>

Reads results from <experiment_directory>/results/, produces two panels:
  - Left:  observable vs (site, time)
  - Right: quench spectral function (QSF) vs (momentum, energy)

Saves <experiment_directory>/figures/qsf.pdf and qsf.png.
"""

import json
import sys
from pathlib import Path

import cmcrameri.cm as cmc
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import colors

from spectroscopy.fourier import compute_fourier_transform, preprocess_for_fourier
from spectroscopy.hardware_utils import process_observations, read_observation_data


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


def main():
    if len(sys.argv) != 2:
        print("Usage: python plot_results.py <experiment_directory>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    res_dir = exp_dir / "results"
    fig_dir = exp_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    with open(res_dir / "experiment_information.json") as f:
        info = json.load(f)

    L = info["L"]
    total_time = info["total_time"]
    slices = info["slices"]
    obs_type = info.get("observable_type", "pauli") or "pauli"

    time_resolution = total_time / (slices - 1)
    times = np.linspace(0.0, total_time, slices, endpoint=True)

    obs_list = read_observation_data(str(exp_dir), time_resolution, slices, L, raw=False)
    obs_avg = process_observations(obs_list)
    L_eff = obs_avg.shape[1]

    baseline = 0.5 if obs_type.startswith("projector") else 0.0
    obs_for_ft = preprocess_for_fourier(obs_avg, baseline=baseline)
    energies, qsf = compute_fourier_transform(obs_for_ft, total_time, slices, L_eff)

    momenta = np.linspace(-np.pi, np.pi, L_eff, endpoint=False)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4), layout="constrained")

    # Left: observable in (site, time)
    ax = axes[0]
    vmin_dyn, vmax_dyn = obs_avg.min(), obs_avg.max()
    dyn_cmap = _zero_white_linear_cmap(cmc.vik, vmin_dyn, vmax_dyn)
    dyn_norm = colors.Normalize(vmin=vmin_dyn, vmax=vmax_dyn)
    im = ax.imshow(
        obs_avg,
        aspect="auto",
        origin="lower",
        extent=[0, L_eff, times[0], times[-1]],
        cmap=dyn_cmap,
        norm=dyn_norm,
    )
    fig.colorbar(im, ax=ax).set_label(r"$G(r, t)$")
    ax.set_xlabel(r"$r$")
    ax.set_ylabel(r"$tJ$")
    # Right: QSF in (momentum, energy)
    ax = axes[1]
    im = ax.imshow(
        qsf,
        aspect="auto",
        origin="lower",
        extent=[momenta[0], momenta[-1], energies[0], energies[-1]],
        cmap=cmc.oslo_r,
        vmin=0,
        vmax=1,
    )
    ax.set_xticks([-np.pi, -np.pi / 2, 0, np.pi / 2, np.pi])
    ax.set_xticklabels([r"$-\pi$", r"$-\pi/2$", r"$0$", r"$\pi/2$", r"$\pi$"])
    ax.set_xlabel(r"$k$")
    ax.set_ylabel(r"$\omega/J$")
    fig.colorbar(im, ax=ax).set_label(r"$G(k, \omega)$")

    for ext, kw in [(".pdf", {}), (".png", {"dpi": 150})]:
        path = fig_dir / ("qsf" + ext)
        fig.savefig(path, **kw)
        print(f"Saved {path}")


if __name__ == "__main__":
    main()
