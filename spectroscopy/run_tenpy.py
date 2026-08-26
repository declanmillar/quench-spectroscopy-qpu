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


import datetime as dt
import os
import platform
import sys
import time
from pathlib import Path

import numpy as np
from tenpy import MPS, Array
from tenpy.algorithms import tdvp
from tenpy.models import SpinChain
from tqdm import tqdm

from spectroscopy.utils import atomic_json_dump, compute_energy_density, get_git_commit

# Enable OpenMP/MKL threading for parallel tensor operations
# Set to number of physical cores for best performance
num_threads = os.environ.get("OMP_NUM_THREADS", "10")
os.environ["OMP_NUM_THREADS"] = num_threads
os.environ["MKL_NUM_THREADS"] = num_threads
os.environ["OPENBLAS_NUM_THREADS"] = num_threads
os.environ["NUMEXPR_NUM_THREADS"] = num_threads


def tenpy_measure(
    psi: MPS,
    meas_axis: str,
    obs_type: str,
    proj_sign: int,
) -> np.ndarray:
    """Measure the requested observable on the current MPS psi; return site/bond-resolved array."""
    # Single-site Pauli or projector
    if obs_type in ("pauli", "projector"):
        obs = tenpy_observable_name(meas_axis)
        # TODO multiplying by two here means we're really measuring sigma not S.
        vals = 2.0 * np.asarray(psi.expectation_value(obs))
        if obs_type == "projector":
            return 0.5 * (1.0 + proj_sign * vals)
        return vals

    if obs_type == "staggered":
        obs = tenpy_observable_name(meas_axis)
        L = psi.L
        # TODO multiplying by two here means we're really measuring sigma not S.
        staggered = np.array([psi.expectation_value_term([(obs, i)]) * (-1) ** i for i in range(L)])
        return staggered

    if obs_type == "pauli2":
        # bond-centered adjacent correlator <sigma_a(i) sigma_a(i+1)>,
        # returned as an array of length L-1
        op = tenpy_observable_name(meas_axis)
        L = psi.L
        SaSa = np.array([psi.expectation_value_term([(op, i), (op, i + 1)]) for i in range(L - 1)])
        # TenPy has S = sigma/2, so sigma*sigma = 4 * S*S
        return 4.0 * SaSa

    if obs_type == "Dx":
        """
        Domain-wall (kink) density on each bond:
            D_i = (1 + Z_i Z_{i+1})/2
        Returns array of length L-1.
        """
        L = psi.L
        # TenPy stores spin operators Sz = sigma_z/2
        SzSz = np.array(
            [psi.expectation_value_term([("Sz", i), ("Sz", i + 1)]) for i in range(L - 1)]
        )
        # Z_i Z_{i+1} = 4 SzSz, so D = (1 - 4 SzSz)/2
        return 0.5 * (1.0 + 4.0 * SzSz)

    if obs_type == "projector2":
        if meas_axis != "sz":
            raise ValueError("projector2 currently implemented for sz only in TenPy")
        Sz = np.asarray(psi.expectation_value("Sz"))
        L = Sz.shape[0]
        SzSz = np.array(
            [psi.expectation_value_term([("Sz", i), ("Sz", i + 1)]) for i in range(L - 1)]
        )
        # P↑↑(j, j+1)
        return 0.25 * (1.0 + 2 * Sz[:-1] + 2 * Sz[1:] + 4 * SzSz)

    if obs_type == "projector3":
        if meas_axis != "sz":
            raise ValueError("projector3 currently implemented for sz only in TenPy")
        Sz = np.asarray(psi.expectation_value("Sz"))
        L = Sz.shape[0]
        SzSz = np.array(
            [psi.expectation_value_term([("Sz", i), ("Sz", i + 1)]) for i in range(L - 1)]
        )
        SzSz2 = np.array(
            [psi.expectation_value_term([("Sz", i), ("Sz", i + 2)]) for i in range(L - 2)]
        )
        SzSzSz = np.array(
            [
                psi.expectation_value_term([("Sz", i - 1), ("Sz", i), ("Sz", i + 1)])
                for i in range(1, L - 1)
            ]
        )

        a, b, c = Sz[:-2], Sz[1:-1], Sz[2:]
        ab, bc = SzSz[:-1], SzSz[1:]
        ac = SzSz2

        # P↑↑↑(j-1,j,j+1)
        return 0.125 * (1.0 + 2.0 * (a + b + c) + 4.0 * (ab + bc + ac) + 8.0 * SzSzSz)

    if obs_type == "flip1":
        Sx = np.array([psi.expectation_value_term([("Sx", i)]) for i in range(psi.L)])
        return 2.0 * Sx

    if obs_type == "flip2":
        # bond-centered, length L-1
        SxSx = np.array(
            [psi.expectation_value_term([("Sx", i), ("Sx", i + 1)]) for i in range(psi.L - 1)]
        )
        SySy = np.array(
            [psi.expectation_value_term([("Sy", i), ("Sy", i + 1)]) for i in range(psi.L - 1)]
        )
        # Proportional to S+S+ + S- S-
        return 2.0 * (SxSx - SySy)

    if obs_type == "flip2_ortho":
        # Orthogonal quadrature to flip2:
        #   Q_I ∝ Sx_i Sy_{i+1} + Sy_i Sx_{i+1}   (Pauli: XY + YX)
        L = psi.L
        SxSy = np.array(
            [psi.expectation_value_term([("Sx", i), ("Sy", i + 1)]) for i in range(L - 1)]
        )
        SySx = np.array(
            [psi.expectation_value_term([("Sy", i), ("Sx", i + 1)]) for i in range(L - 1)]
        )
        # Keep normalization consistent with flip2:
        # flip2 returns 2*(SxSx - SySy), so here return 2*(SxSy + SySx)
        return 2.0 * (SxSy + SySx)

    if obs_type == "flip2_complex":
        # Complex pair field ~ (XX-YY) + i(XY+YX)
        L = psi.L

        SxSx = np.array(
            [psi.expectation_value_term([("Sx", i), ("Sx", i + 1)]) for i in range(L - 1)]
        )
        SySy = np.array(
            [psi.expectation_value_term([("Sy", i), ("Sy", i + 1)]) for i in range(L - 1)]
        )
        SxSy = np.array(
            [psi.expectation_value_term([("Sx", i), ("Sy", i + 1)]) for i in range(L - 1)]
        )
        SySx = np.array(
            [psi.expectation_value_term([("Sy", i), ("Sx", i + 1)]) for i in range(L - 1)]
        )

        real_part = 2.0 * (SxSx - SySy)  # matches flip2
        imag_part = 2.0 * (SxSy + SySx)  # orthogonal quadrature
        return real_part - 1j * imag_part

    if obs_type == "flip3":
        L = psi.L
        SxSxSx = np.array(
            [
                psi.expectation_value_term([("Sx", i), ("Sx", i + 1), ("Sx", i + 2)])
                for i in range(L - 2)
            ]
        )
        SxSySy = np.array(
            [
                psi.expectation_value_term([("Sx", i), ("Sy", i + 1), ("Sy", i + 2)])
                for i in range(L - 2)
            ]
        )
        SySxSy = np.array(
            [
                psi.expectation_value_term([("Sy", i), ("Sx", i + 1), ("Sy", i + 2)])
                for i in range(L - 2)
            ]
        )
        SySySx = np.array(
            [
                psi.expectation_value_term([("Sy", i), ("Sy", i + 1), ("Sx", i + 2)])
                for i in range(L - 2)
            ]
        )
        # Proportional to S+S+S+ + S- S- S-
        return SxSxSx - (SxSySy + SySxSy + SySySx)

    if obs_type == "XYbond":
        # bond-centered, length L-1
        SxSx = np.array(
            [psi.expectation_value_term([("Sx", i), ("Sx", i + 1)]) for i in range(psi.L - 1)]
        )
        SySy = np.array(
            [psi.expectation_value_term([("Sy", i), ("Sy", i + 1)]) for i in range(psi.L - 1)]
        )
        return 2.0 * (SxSx + SySy)

    raise ValueError(f"Unknown obs_type: {obs_type}")


def tenpy_observable_name(meas_axis: str) -> str:
    # TenPy spin-1/2 operators are normalized as S_k = sigma_k / 2
    return {"sx": "Sx", "sy": "Sy", "sz": "Sz"}[meas_axis]


def pauli_rotation_matrix(axis: str, theta: float) -> Array:
    # TenPy expects labels ["p","p*"]
    c = np.cos(theta / 2.0)
    s = np.sin(theta / 2.0)
    if axis == "rx":
        operator = np.array([[c, -1j * s], [-1j * s, c]])
    elif axis == "ry":
        operator = np.array([[c, -s], [s, c]])
    elif axis == "rz":
        operator = np.array([[np.exp(-1j * theta / 2), 0.0], [0.0, np.exp(1j * theta / 2)]])
    else:
        raise ValueError("axis must be rx|ry|rz")
    return Array.from_ndarray_trivial(operator, labels=["p", "p*"])


def projector_sign(sign: str) -> int:
    return +1 if sign == "plus" else -1


def run_tenpy(
    args,
    result_dir: Path,
    fig_dir: Path,
    model: SpinChain,
    ground_state: MPS,
    quench_sites,
):
    energy_density = compute_energy_density(model, ground_state)
    print(f"[Info] Ground state energy density: {energy_density}")

    # Print threading configuration
    print("[Info] Threading configuration:")
    print(f"       OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'not set')}")
    print(f"       OPENBLAS_NUM_THREADS: {os.environ.get('OPENBLAS_NUM_THREADS', 'not set')}")

    # Quick BLAS performance benchmark
    print("[Info] Running quick BLAS benchmark...")
    bench_size = 1000
    A_bench = np.random.rand(bench_size, bench_size)
    B_bench = np.random.rand(bench_size, bench_size)
    bench_start = time.perf_counter()
    _ = np.dot(A_bench, B_bench)
    bench_time = time.perf_counter() - bench_start
    bench_gflops = (2 * bench_size**3) / bench_time / 1e9
    print(f"       Matrix size: {bench_size}x{bench_size}")
    print(f"       Time: {bench_time:.3f} seconds")
    print(f"       GFLOPS: {bench_gflops:.2f}")
    if bench_gflops > 20:
        print("       ✓ Excellent performance (optimized BLAS with threading)")
    elif bench_gflops > 10:
        print("       ✓ Good performance (optimized BLAS)")
    else:
        print("       ⚠ Suboptimal performance - consider enabling threading")

    # Start timing: total simulation time
    time_start_total = time.perf_counter()

    # Start from ground state, apply quench once, then evolve once and measure at each slice.
    psi = ground_state.copy()

    # Apply local quench on selected sites
    rot = pauli_rotation_matrix(args.quench_op, args.quench_angle)
    for s in quench_sites:
        psi.apply_local_op(i=s, op=rot, unitary=True)
    psi.canonical_form()

    tdvp_params = {
        "start_time": 0.0,
        "trunc_params": {"chi_max": args.max_chi, "trunc_cut": args.tdvp_trunc_cut},
    }
    tdvp_engine = tdvp.TwoSiteTDVPEngine(psi, model, tdvp_params)

    # Times and observable
    times = np.linspace(0.0, args.total_time, args.slices, endpoint=True)

    axis_for_proj = args.proj_axis or args.meas_axis
    proj_s = projector_sign(args.proj_sign)

    if args.slices < 2:
        raise ValueError("args.slices must be >= 2")

    # Interpret args.num_trotter_steps as number of micro-steps per slice interval.
    micro_steps_per_slice = args.num_trotter_steps
    dt_micro = args.total_time / ((args.slices - 1) * micro_steps_per_slice)

    # Measure along the trajectory: t=0 plus after each slice interval.
    obs_list = []
    chi_max_per_slice = []

    # Start timing: evolution only (excluding initial setup)
    time_start_evolution = time.perf_counter()

    obs_list.append(tenpy_measure(psi, axis_for_proj, args.obs_type, proj_s))
    chi_max_per_slice.append(int(max(psi.chi)))

    for _ in tqdm(range(1, args.slices)):
        tdvp_engine.evolve(N_steps=micro_steps_per_slice, dt=dt_micro)
        obs_list.append(tenpy_measure(psi, axis_for_proj, args.obs_type, proj_s))
        chi_max_per_slice.append(int(max(psi.chi)))

    # End timing
    time_end_evolution = time.perf_counter()
    time_end_total = time.perf_counter()

    # Calculate timing metrics
    evolution_time = time_end_evolution - time_start_evolution
    total_time = time_end_total - time_start_total

    print(f"[Timing] Evolution time: {evolution_time:.3f} seconds")
    print(f"[Timing] Total simulation time: {total_time:.3f} seconds")

    obs_values = np.stack(obs_list, axis=0)

    np.save(result_dir / "tenpy_times.npy", times)
    np.save(result_dir / "tenpy_observables.npy", obs_values)

    experiment_info = {
        "mode": "tenpy",
        "L": args.L,
        "total_time": args.total_time,
        "slices": args.slices,
        "num_trotter_steps": args.num_trotter_steps,
        "Jx": args.Jx,
        "Jy": args.Jy,
        "Jz": args.Jz,
        "hx": args.hx,
        "hy": args.hy,
        "hz": args.hz,
        "staggered": args.staggered,
        "break_symmetries": args.break_symmetries,
        "product_state": args.product_state,
        "initial_state": args.init_state,
        "ground_state_energy_density": energy_density,
        "dmrg": {
            "chi_max": args.dmrg_chi,
            "svd_min": args.dmrg_svd_min,
            "max_E_err": args.dmrg_max_e_err,
        },
        "quench": {
            "op": args.quench_op,
            "angle": args.quench_angle,
            "sites": quench_sites,
        },
        "observable_type": args.obs_type,
        "meas_axis": args.meas_axis,
        "projector": (
            {"axis": axis_for_proj, "sign": args.proj_sign}
            if args.obs_type == "projector"
            else (
                {"axis": axis_for_proj, "order": 2}
                if args.obs_type == "projector2"
                else (
                    {"axis": axis_for_proj, "order": 3} if args.obs_type == "projector3" else None
                )
            )
        ),
        "git_commit": get_git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
        "tdvp": {"chi_max": args.max_chi, "trunc_cut": args.tdvp_trunc_cut},
        "tdvp_dt_micro": dt_micro,
        "tdvp_micro_steps_per_slice": micro_steps_per_slice,
        "chi_max_per_slice": chi_max_per_slice,
        "timing": {
            "evolution_time_seconds": evolution_time,
            "total_simulation_time_seconds": total_time,
            "time_per_slice_seconds": evolution_time / (args.slices - 1)
            if args.slices > 1
            else 0.0,
        },
    }
    atomic_json_dump(experiment_info, result_dir / "experiment_information.json")
