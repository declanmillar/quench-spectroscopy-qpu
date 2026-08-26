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

"""Utility functions."""

import argparse
import datetime as dt
import json
import os
import subprocess
import tempfile
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from qiskit import QuantumCircuit
from tenpy import MPS, SpinChain
from tenpy.networks.site import SpinHalfSite, SpinSite


def multi_qubit_gate_depth(qc: QuantumCircuit) -> int:
    """
    Return the multi-qubit gate depth.

    When the circuit has been transpiled for IBM Quantum hardware
    this will be equivalent to the CNOT depth.

    Args:
        qc: The quantum circuit to calculate the depth of.

    Returns:
        The multi-qubit depth of the circuit.
    """
    return qc.depth(filter_function=lambda instr: len(instr.qubits) > 1)


def atomic_json_dump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, dir=str(path.parent)) as tmp:
        json.dump(obj, tmp, indent=2)
        tmp.flush()
        os.fsync(tmp.fileno())
        tmp_name = tmp.name
    Path(tmp_name).replace(path)


def get_git_commit() -> str:
    try:
        return (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except OSError, subprocess.CalledProcessError:
        return "unknown"


def figure_save(figpath: Path, fig=None):
    if fig is None:
        plt.savefig(figpath)
        plt.close()
    else:
        fig.savefig(figpath)
        plt.close(fig)


def parse_spins(spec: str, L: int, center: int) -> list[int]:
    """
    Parse spin-selection strings.
      - 'center' -> [center]
      - 'all'    -> list(range(L))
      - comma list of ints: '0,5,10'
      - half-open ranges: '10:20' (-> 10..19)
      - mix allowed: 'center,0,10:15'
    """
    spec = spec.strip().lower()
    if spec == "all":
        return list(range(L))
    if spec == "center":
        return [center]
    out = []
    for tok in filter(None, [t.strip() for t in spec.split(",")]):
        if tok == "center":
            out.append(center)
            continue
        if ":" in tok:
            a, b = tok.split(":", 1)
            a, b = int(a), int(b)
            out.extend(range(a, b))
        else:
            out.append(int(tok))
    # clamp, unique, sorted
    out = sorted({i for i in out if 0 <= i < L})
    if not out:
        raise ValueError(f"No valid spins parsed from '{spec}' within [0, {L - 1}]")
    return out


def break_model_symmetry(
    L: int,
    Jx: float,
    hz: float,
    conserve: str | None = None,
    axis: str = "Sy",
    strength: float = 0.1,
    site: int = 0,
    hx: float = 0.0,
    hy: float = 0.0,
    staggered: bool = False,
) -> SpinChain:
    """Return a TenPy SpinChain used only for GS preparation."""
    # Build field arrays (uniform or staggered)
    if staggered:
        hx_array = np.array([(-1) ** i * hx for i in range(L)])
        hy_array = np.array([(-1) ** i * hy for i in range(L)])
        hz_array = np.array([(-1) ** i * hz for i in range(L)])
    else:
        hx_array = hx
        hy_array = hy
        hz_array = hz

    gs_model = SpinChain(
        {
            "L": L,
            "Jx": Jx,
            "Jy": 0.0,
            "Jz": 0.0,
            "hx": hx_array,
            "hy": hy_array,
            "hz": hz_array,
            "bc_MPS": "finite",
            "conserve": conserve,
        }
    )
    if axis not in {"Sx", "Sy", "Sz"}:
        raise ValueError("axis must be Sx, Sy, or Sz")
    if not (0 <= site < L):
        raise ValueError(f"site {site} out of range 0..{L - 1}")

    gs_model.manually_call_init_H = True
    gs_model.add_onsite(strength, site, axis)
    gs_model.init_H_from_terms()
    return gs_model


def make_paths_and_labels(args: argparse.Namespace, quench_sites: list[int]) -> tuple[Path, Path]:
    """Create experiment directory tree.

    Args:
        args: Parsed command-line arguments containing model parameters,
            quench info, and measurement axis.
        quench_sites: Indices of the spins to which the quench is applied.

    Returns:
        A tuple containing:
            - result_dir: Directory path for numerical results and metadata.
            - fig_dir: Directory path for generated figures.
    """
    qlabel = f"{args.quench_op}_{args.quench_angle:.3f}_S{('-'.join(map(str, quench_sites)))}"
    olabel = f"{args.obs_type}_{args.meas_axis}"
    stamp = dt.datetime.now(tz=dt.UTC).strftime("%Y_%m_%d_%H_%M_%S")

    exp_root = Path(
        f"experiments/"
        f"{stamp}_L_{args.L}_Jx_{args.Jx}_Jy_{args.Jy}_Jz_{args.Jz}_hz_{args.hz}"
        f"__Q_{qlabel}__O_{olabel}"
    )

    result_dir = exp_root / "results"
    fig_dir = exp_root / "figures"
    exp_root.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    return result_dir, fig_dir


def compute_energy_density(model: SpinChain, psi: MPS) -> float:
    """Compute the energy density of a given MPS state for a SpinChain model."""
    H_mpo = model.H_MPO

    E_tot = H_mpo.expectation_value(psi)

    L = model.lat.N_sites
    e_density = E_tot / L
    # Convert to float to ensure JSON serialization works
    # (TenPy may return complex128 even for real values)
    return float(np.real(e_density))


def qiskit_to_tenpy_mps(qiskit_mps: tuple, return_form: str = "SpinSite") -> MPS:
    """Convert a Qiskit Aer MPS to a TeNPy MPS.

    Args:
        qiskit_mps: Raw MPS produced by ``QiskitAerMPS``, i.e. a tuple
            ``(gamma, lambda)`` where *gamma* is a list of ``(A_0, A_1)``
            pairs and *lambda* is a list of singular-value vectors.
        return_form: TeNPy site type to use - ``"SpinSite"`` (default) or
            ``"SpinHalfSite"``.

    Returns:
        A TeNPy ``MPS`` object.
    """
    gam, lam = qiskit_mps
    # Stack each (gamma_up, gamma_dn) pair into a (2, bond_L, bond_R) tensor
    # and absorb the right-adjacent Lambda into all but the last site.
    tensors = [
        np.stack((gam[n][0], gam[n][1]))
        * (np.expand_dims(lam[n], (0, 1)) if n < len(gam) - 1 else 1)
        for n in range(len(gam))
    ]

    if return_form == "SpinSite":
        sites = [SpinSite(conserve=None)] * len(tensors)
        # SpinSite uses the opposite basis ordering to Qiskit - flip physical axis.
        tensors = [np.flip(t, axis=0) for t in tensors]
    elif return_form == "SpinHalfSite":
        sites = [SpinHalfSite(conserve=None)] * len(tensors)
    else:
        raise ValueError(
            f"Invalid return_form: {return_form!r}. Must be 'SpinSite' or 'SpinHalfSite'."
        )

    return MPS.from_Bflat(sites, tensors, SVs=None)
