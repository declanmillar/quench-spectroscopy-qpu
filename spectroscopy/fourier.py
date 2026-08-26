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
Function to perform a 2D Fourier transform.
"""

import numpy as np
from scipy import fftpack
from scipy.signal import windows


def preprocess_for_fourier(
    obs_data: np.ndarray,
    baseline: float = 0.0,
    remove_time_mean: bool = False,
    remove_spatial_mean: bool = False,
    use_time_derivative: bool = False,
    times: np.ndarray | None = None,
) -> np.ndarray:
    """
    Preprocess observable data before Fourier transform to suppress DC and k≈0 components.

    Parameters
    ----------
    obs_data : np.ndarray
        Observable data with shape (time_steps, spatial_sites).
    baseline : float, optional
        Baseline value to subtract (e.g., 0.5 for projector observables). Default is 0.0.
    remove_time_mean : bool, optional
        If True, remove per-site/bond time mean to kill large diagonal/DC component. Default is False.
    remove_spatial_mean : bool, optional
        If True, remove spatial mean at each time to kill strong k=0 component. Default is False.
    use_time_derivative : bool, optional
        If True, take time derivative to further suppress DC leakage. Default is False.
    times : np.ndarray, optional
        Time array, required if use_time_derivative is True.

    Returns
    -------
    np.ndarray
        Preprocessed observable data ready for Fourier transform.
    """
    obs_for_ft = obs_data - baseline

    if remove_time_mean:
        # Remove per-site/bond time mean: kills large diagonal/DC component
        obs_for_ft = obs_for_ft - obs_for_ft.mean(axis=0, keepdims=True)

    if remove_spatial_mean:
        # Remove spatial mean at each time: kills strong k=0 component
        obs_for_ft = obs_for_ft - obs_for_ft.mean(axis=1, keepdims=True)

    if use_time_derivative:
        # Take time derivative to further suppress DC leakage
        if times is None:
            raise ValueError("times array must be provided when use_time_derivative=True")
        dt = times[1] - times[0]
        obs_for_ft = np.gradient(obs_for_ft, dt, axis=0)

    return obs_for_ft


def compute_fourier_transform(
    obs_data: np.ndarray,
    total_time: float,
    steps: int,
    L: int,
    window_type: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Compute the 2D Fourier transform of the observable data."""
    if window_type is None:
        window = np.ones((steps, L), dtype=np.float64)
    elif window_type == "hann":
        window = np.outer(windows.hann(steps), windows.hann(L)).astype(np.float64)
    else:
        raise ValueError(f"Unsupported window type: {window_type}")

    filtered_data = window * obs_data
    ft = np.abs(np.fft.fft2(filtered_data, norm=None))
    ft = np.fft.fftshift(ft)

    time_resolution = total_time / (steps - 1)
    energies = fftpack.fftshift(fftpack.fftfreq(steps) * (2.0 * np.pi / time_resolution))

    return energies, np.abs(ft) / np.max(np.abs(ft))
