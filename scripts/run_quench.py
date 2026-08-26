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

import argparse
import os
import pickle
from pathlib import Path

import numpy as np
from mps_to_circuit import mps_to_circuit
from qiskit import QuantumCircuit
from qiskit.qasm3 import loads as qasm3_loads
from qiskit_addon_aqc_tensor.simulation import tensornetwork_from_circuit
from qiskit_addon_aqc_tensor.simulation.aer import QiskitAerMPS
from qiskit_aer import AerSimulator
from tenpy import MPS
from tenpy.algorithms import dmrg
from tenpy.models import SpinChain

from spectroscopy.run_hardware import run_hardware
from spectroscopy.run_tenpy import run_tenpy
from spectroscopy.utils import (
    break_model_symmetry,
    make_paths_and_labels,
    multi_qubit_gate_depth,
    parse_spins,
    qiskit_to_tenpy_mps,
)


def parse_args():
    p = argparse.ArgumentParser(description="Batch local-quench run (hardware | TenPy)")
    p.add_argument("--mode", choices=["hardware", "tenpy"], default="hardware")

    # Lattice / Hamiltonian (SpinChain convention!)
    p.add_argument("-L", type=int, default=101)
    p.add_argument("--Jx", type=float, default=-1.0)
    p.add_argument("--Jy", type=float, default=-1.0)
    p.add_argument("--Jz", type=float, default=-2.0)
    p.add_argument("--hx", type=float, default=0.0)
    p.add_argument("--hy", type=float, default=0.0)
    p.add_argument("--hz", type=float, default=0.0)
    p.add_argument("--staggered", action="store_true")

    # Quench / measurement
    p.add_argument("-O", "--quench-op", choices=["rx", "ry", "rz"], default="ry")
    p.add_argument("--quench-angle", type=float, default=np.pi / 2)
    p.add_argument(
        "--antisymmetric",
        action="store_true",
        help="Submit both +θ and -θ quenches in single batch for antisymmetric difference",
    )
    p.add_argument("-m", "--quench-sites", type=str, default="center")
    p.add_argument("-M", "--meas-axis", choices=["sx", "sy", "sz"], default="sy")
    p.add_argument(
        "--obs-type",
        choices=[
            "pauli",
            "projector",
            "projector2",
            "projector3",
            "pauli2",
            "flip1",
            "flip2",
            "flip2_ortho",
            "flip2_complex",
            "flip3",
            "Dx",
            "bond",
            "staggered",
            "XYbond",
        ],
        default="pauli",
    )
    p.add_argument("--proj-axis", choices=["sx", "sy", "sz"])
    p.add_argument("--proj-sign", choices=["plus", "minus"], default="plus")

    # Time evolution
    p.add_argument("-T", "--total-time", type=float, default=10.0)
    p.add_argument("--slices", type=int, default=21)
    p.add_argument("--fixed-dt", action="store_true")
    p.add_argument("-N", "--num-trotter-steps", type=int, default=10)

    # Initial state / DMRG
    p.add_argument("--break-symmetries", action="store_true")
    p.add_argument("--product-state", action="store_true")
    p.add_argument(
        "--init-state",
        choices=[
            "polarized_up",
            "polarized_down",
            "neel_up_down",
            "neel_down_up",
            "random",
        ],
        default="polarized_down",
    )
    p.add_argument("--dmrg-chi", type=int, default=64)
    p.add_argument("--dmrg-svd-min", type=float, default=1e-10)
    p.add_argument("--dmrg-max-e-err", type=float, default=1e-10)

    # QASM → MPS initialisation
    p.add_argument("--init-from-qasm", type=str, default=None)
    p.add_argument("--qasm-mps-max-bond", type=int, default=50)
    p.add_argument("--qasm-mps-trunc-threshold", type=float, default=1e-8)
    p.add_argument("--qasm-energy-only", action="store_true")
    p.add_argument("--qasm-compare-dmrg", action="store_true")

    # Pickle → MPS initialisation
    p.add_argument("--init-from-mps", type=str, default=None, help="Path to pickled TenPy MPS file")
    p.add_argument(
        "--pickle-energy-only",
        action="store_true",
        help="Only compute energy of pickled MPS and exit",
    )
    p.add_argument(
        "--pickle-compare-dmrg",
        action="store_true",
        help="Compare pickled MPS energy with DMRG ground state",
    )

    # Hardware-only
    p.add_argument("--backend", type=str, default="ibm_boston")
    p.add_argument("-s", "--shots", type=int, default=10000)
    p.add_argument("--opt-level", type=int, default=2)
    p.add_argument("--seed-transpiler", type=int, default=1234)
    p.add_argument("--zne", action="store_true")
    p.add_argument("--pea", action="store_true")
    p.add_argument("--dd", action="store_false")
    p.add_argument("--pt", action="store_false")
    p.add_argument("--num-processes", type=int, default=os.cpu_count())
    p.add_argument(
        "--mps-truncation-threshold",
        type=float,
        default=1e-5,
        help="Truncation threshold for AerSimulator MPS method",
    )
    p.add_argument(
        "--mps-max-bond-dimension",
        type=int,
        default=128,
        help="Max bond dimension for AerSimulator MPS method",
    )
    # TenPy-only
    p.add_argument("--max-chi", type=int, default=128)
    p.add_argument("--tdvp-trunc-cut", type=float, default=1e-4)
    # Save options
    p.add_argument(
        "--save-gs-mps",
        action="store_true",
        help="Save ground state MPS as pickle file to results directory",
    )

    return p.parse_args()


def load_qiskit_circuit_from_qasm(path: str) -> QuantumCircuit:
    """Load a Qiskit QuantumCircuit from an OpenQASM 3 file."""
    with open(Path(path).expanduser()) as f:
        qc = qasm3_loads(f.read())
    print("[QASM] circuit depth (2q)", qc.depth(lambda x: len(x.qubits) > 1))
    return qc


def qiskit_circuit_to_tenpy_mps(
    qc: QuantumCircuit,
    *,
    mps_trunc_threshold: float,
    mps_max_bond_dimension: int,
) -> MPS:
    """Convert a Qiskit circuit to a TenPy MPS via Aer MPS simulation."""
    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_truncation_threshold=mps_trunc_threshold,
        matrix_product_state_max_bond_dimension=mps_max_bond_dimension,
    )
    qiskit_mps: QiskitAerMPS = tensornetwork_from_circuit(qc, sim)  # ty: ignore[invalid-assignment]
    tenpy_mps = qiskit_to_tenpy_mps((qiskit_mps.gamma, qiskit_mps.lamb))
    n2 = tenpy_mps.overlap(tenpy_mps)  # <psi|psi>
    print("[QASM] norm^2 =", float(np.real(n2)))
    tenpy_mps.canonical_form()
    return tenpy_mps


def build_initial_product_state(init_state: str, L: int) -> list[str]:
    if init_state == "polarized_up":
        return ["up"] * L
    if init_state == "polarized_down":
        return ["down"] * L
    if init_state == "neel_up_down":
        return (["up", "down"] * (L // 2)) + (["up"] if L % 2 else [])
    if init_state == "neel_down_up":
        return (["down", "up"] * (L // 2)) + (["down"] if L % 2 else [])
    if init_state == "random":
        return list(np.random.choice(["up", "down"], size=L))
    raise ValueError(init_state)


def load_tenpy_mps_from_pickle(path):
    """Load a pickled TenPy MPS from file."""
    with open(Path(path).expanduser(), "rb") as f:
        mps = pickle.load(f)
    print(f"[Pickle] Loaded MPS from {path}")
    print(f"[Pickle] MPS length: {mps.L}, max bond dimension: {np.max(mps.chi)}")
    return mps


def load_tenpy_mps_from_qasm(path, *, mps_trunc_threshold, mps_max_bond_dimension):
    with open(Path(path).expanduser()) as f:
        qc = qasm3_loads(f.read())

    print("[QASM] circuit depth", qc.depth(lambda x: len(x.qubits) > 1))

    sim = AerSimulator(
        method="matrix_product_state",
        matrix_product_state_truncation_threshold=mps_trunc_threshold,
        matrix_product_state_max_bond_dimension=mps_max_bond_dimension,
    )
    qiskit_mps: QiskitAerMPS = tensornetwork_from_circuit(qc, sim)  # ty: ignore[invalid-assignment]
    tenpy_mps = qiskit_to_tenpy_mps((qiskit_mps.gamma, qiskit_mps.lamb))

    n2 = tenpy_mps.overlap(tenpy_mps)  # <psi|psi>
    print("norm^2 =", np.real(n2))

    tenpy_mps.canonical_form()
    return tenpy_mps


def build_model(args):
    # Build field arrays (uniform or staggered)
    if args.staggered:
        # Staggered pattern: alternating +h/-h on even/odd sites
        hx_array = np.array([(-1) ** i * args.hx for i in range(args.L)])
        hy_array = np.array([(-1) ** i * args.hy for i in range(args.L)])
        hz_array = np.array([(-1) ** i * args.hz for i in range(args.L)])
    else:
        # Uniform field
        hx_array = args.hx
        hy_array = args.hy
        hz_array = args.hz

    return SpinChain(
        {
            "L": args.L,
            "Jx": args.Jx,
            "Jy": args.Jy,
            "Jz": args.Jz,
            "hx": hx_array,
            "hy": hy_array,
            "hz": hz_array,
            "bc_MPS": "finite",
            "conserve": None,
        }
    )


def energy_of_state(model, psi):
    return float(np.real(model.calc_H_MPO().expectation_value(psi)))


def build_ground_state_via_dmrg(args, model, result_dir=None):
    print("[DMRG] Computing MPS ground state...")
    product = build_initial_product_state(args.init_state, model.lat.N_sites)
    psi0 = MPS.from_product_state(model.lat.mps_sites(), product, bc=model.lat.bc_MPS)

    eng = dmrg.SingleSiteDMRGEngine(
        psi0,
        model,
        {
            "mixer": True,
            "trunc_params": {
                "chi_max": args.dmrg_chi,
                "svd_min": args.dmrg_svd_min,
            },
            "max_E_err": args.dmrg_max_e_err,
            "combine": True,
        },
    )
    _, gs = eng.run()

    print(f"[DMRG] Ground state MPS max bond dimension {np.max(gs.chi)}")

    # Save ground state MPS to pickle file if result_dir is provided and save option is enabled
    if result_dir is not None and args.save_gs_mps:
        gs_pickle_path = result_dir / "ground_state_mps.pkl"
        with open(gs_pickle_path, "wb") as f:
            pickle.dump(gs, f)
        print(f"[DMRG] Ground state MPS saved to {gs_pickle_path}")
        raise SystemExit(0)

    return gs


def build_model_and_initial_state(args, result_dir=None):
    """
    Returns:
      model: SpinChain
      psi0: Optional[tenpy.MPS]  (may be None in hardware mode with --init-from-qasm)
      qc_init: Optional[qiskit.QuantumCircuit]
    """
    model = build_model(args)

    qc_init = None
    psi = None

    # Load from pickled MPS if specified
    if args.init_from_mps:
        psi = load_tenpy_mps_from_pickle(args.init_from_mps)

        e = energy_of_state(model, psi)
        print(f"[Pickle] <psi|H|psi> = {e:.12f}")

        if args.pickle_compare_dmrg:
            gs = build_ground_state_via_dmrg(args, model, result_dir)
            e_gs = energy_of_state(model, gs)
            print(f"[DMRG] <gs|H|gs> = {e_gs:.12f}")
            print(f"[ΔE] = {e - e_gs:.3e}")

            # Calculate fidelity
            overlap = psi.overlap(gs)
            fidelity = abs(overlap) ** 2
            print(f"[Pickle] Fidelity with DMRG ground state: {fidelity:.6f}")

        if args.pickle_energy_only:
            raise SystemExit(0)

        return model, psi, None

    if args.init_from_qasm:
        qc_init = load_qiskit_circuit_from_qasm(args.init_from_qasm)

        qc_init_stripped = qc_init.remove_final_measurements(inplace=False)
        assert qc_init_stripped is not None
        qc_init = qc_init_stripped

        # Only build TenPy MPS if needed (TenPy mode or explicit energy/DMRG checks)
        need_tenpy_mps = (args.mode == "tenpy") or args.qasm_energy_only or args.qasm_compare_dmrg
        if need_tenpy_mps:
            psi = qiskit_circuit_to_tenpy_mps(
                qc_init,
                mps_trunc_threshold=args.qasm_mps_trunc_threshold,
                mps_max_bond_dimension=args.qasm_mps_max_bond,
            )

            e = energy_of_state(model, psi)
            print(f"[QASM] <psi|H|psi> = {e:.12f}")

            if args.qasm_compare_dmrg:
                gs = build_ground_state_via_dmrg(args, model, result_dir)
                e_gs = energy_of_state(model, gs)
                print(f"[DMRG] <gs|H|gs> = {e_gs:.12f}")
                print(f"[ΔE] = {e - e_gs:.3e}")

            if args.qasm_energy_only:
                raise SystemExit(0)

        return model, psi, qc_init

    # Non-QASM init: build TenPy MPS
    if args.break_symmetries:
        gs_model = break_model_symmetry(
            L=args.L,
            Jx=args.Jx,
            hx=args.hx,
            hy=args.hy,
            hz=args.hz,
            staggered=args.staggered,
            conserve=None,
            strength=0.1,
            axis="Sx",
            site=0,
        )
    else:
        gs_model = model

    product = build_initial_product_state(args.init_state, model.lat.N_sites)
    psi_product = MPS.from_product_state(model.lat.mps_sites(), product, bc=model.lat.bc_MPS)

    # Always compute ground state (needed for fidelity comparison when using product state)
    gs = build_ground_state_via_dmrg(args, gs_model, result_dir)

    if args.product_state:
        # Calculate fidelity between product state and ground state
        overlap = psi_product.overlap(gs)
        fidelity = abs(overlap) ** 2
        print(f"[Product State] Fidelity with ground state: {fidelity:.6f}")

        return model, psi_product, None

    return model, gs, None


def main():
    args = parse_args()
    center = args.L // 2
    quench_sites = parse_spins(args.quench_sites, args.L, center)
    result_dir, fig_dir = make_paths_and_labels(args, quench_sites)

    model, psi0, qc_init = build_model_and_initial_state(args, result_dir)

    if args.mode == "tenpy":
        if psi0 is None:
            raise ValueError(
                "TenPy mode requires a TenPy MPS initial state, but psi0 is None. "
                "If you passed --init-from-qasm, MPS conversion was skipped unexpectedly."
            )
        run_tenpy(args, result_dir, fig_dir, model, psi0, quench_sites)
        return

    # ---------------- Hardware mode ----------------
    # We want an initial circuit qc0.
    if qc_init is not None:
        qc0 = qc_init
    else:
        # Map TenPy MPS -> circuit here (moved out of run_hardware)

        if psi0 is None:
            raise ValueError("Hardware mode requires either --init-from-qasm or a TenPy MPS psi0.")

        # Build MPS tensors, auto-detecting TenPy convention
        Sz_op = psi0.sites[0].get_op("Sz").to_ndarray()
        flip_physical = Sz_op[0, 0] < 0  # True for SpinSite (opposite convention)
        mps_arrays = []
        for i in range(args.L):
            arr = psi0.get_B(i, form="A").itranspose(["vL", "p", "vR"]).to_ndarray()
            if flip_physical:
                arr = arr[:, ::-1, :]
            mps_arrays.append(arr)

        qc0 = mps_to_circuit(mps_arrays, method="approximate", shape="lpr", num_layers=1)
        print(f"[Prep] GS CX-depth (ansatz): {multi_qubit_gate_depth(qc0)}")

    # Updated signature: pass qc0 in
    run_hardware(args, result_dir, fig_dir, model, psi0, qc0, quench_sites)


if __name__ == "__main__":
    main()
