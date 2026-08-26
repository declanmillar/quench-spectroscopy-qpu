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
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager
from qiskit.visualization import plot_circuit_layout, plot_error_map
from qiskit_aer.primitives import EstimatorV2 as AerEstimatorV2
from qiskit_ibm_runtime import (
    Batch,
    EstimatorOptions,
    QiskitRuntimeService,
)
from qiskit_ibm_runtime import (
    EstimatorV2 as RuntimeEstimatorV2,
)
from qiskit_ibm_runtime.options import (
    DynamicalDecouplingOptions,
    ResilienceOptionsV2,
    TwirlingOptions,
)
from tenpy import MPS, SpinChain

from spectroscopy.evolution import build_xyz_evolution_circuits
from spectroscopy.fixed_aer_simulator import FixedAerSimulator as AerSimulator
from spectroscopy.hardware_utils import (
    build_observables,
    get_qubit_assignment_from_EPLG,
    save_result,
)
from spectroscopy.utils import (
    atomic_json_dump,
    compute_energy_density,
    figure_save,
    get_git_commit,
)


def run_hardware(
    args,
    result_dir: Path,
    fig_dir: Path,
    model: SpinChain,
    ground_state: MPS | None,
    qc0,
    quench_sites,
):
    if ground_state is not None:
        print(f"Sz = {ground_state.sites[0].get_op('Sz').to_ndarray()}")
        energy_density = compute_energy_density(model, ground_state)
        print(f"[Info] Ground state energy density: {energy_density}")
    else:
        energy_density = None

    # Decide whether we are using local Aer or IBM Runtime
    use_aer = args.backend == "aer_mps"

    if qc0 is None:
        raise ValueError("run_hardware expected a prepared initial circuit qc0, got None.")

    # Check if antisymmetric mode is enabled
    antisymmetric = getattr(args, "antisymmetric", False)

    if antisymmetric:
        # Create circuits for both +θ and -θ quenches
        print("[Antisymmetric] Creating circuits for both +θ and -θ quenches")

        # +θ quench circuits
        qc0_plus = qc0.copy()
        for site in quench_sites:
            getattr(qc0_plus, args.quench_op)(args.quench_angle, site)

        circuits_plus, circuit_info_plus = build_xyz_evolution_circuits(
            qc0_plus,
            args.total_time,
            args.slices,
            args.fixed_dt,
            args.num_trotter_steps,
            args.Jx,
            args.Jy,
            args.Jz,
            args.hx,
            args.hy,
            args.hz,
            args.staggered,
        )

        # -θ quench circuits
        qc0_minus = qc0.copy()
        for site in quench_sites:
            getattr(qc0_minus, args.quench_op)(-args.quench_angle, site)

        circuits_minus, circuit_info_minus = build_xyz_evolution_circuits(
            qc0_minus,
            args.total_time,
            args.slices,
            args.fixed_dt,
            args.num_trotter_steps,
            args.Jx,
            args.Jy,
            args.Jz,
            args.hx,
            args.hy,
            args.hz,
            args.staggered,
        )

        # Rename circuits to distinguish +θ and -θ
        for i, c in enumerate(circuits_plus):
            c.name = f"{c.name}_plus"
            circuit_info_plus[i]["quench_sign"] = "plus"
        for i, c in enumerate(circuits_minus):
            c.name = f"{c.name}_minus"
            circuit_info_minus[i]["quench_sign"] = "minus"

        # Combine circuits and info
        circuits = circuits_plus + circuits_minus
        circuit_info = circuit_info_plus + circuit_info_minus

    else:
        # Standard single quench
        for site in quench_sites:
            getattr(qc0, args.quench_op)(args.quench_angle, site)

        circuits, circuit_info = build_xyz_evolution_circuits(
            qc0,
            args.total_time,
            args.slices,
            args.fixed_dt,
            args.num_trotter_steps,
            args.Jx,
            args.Jy,
            args.Jz,
            args.hx,
            args.hy,
            args.hz,
            args.staggered,
        )

    # Observables: per-site single-axis Pauli (assembled via helper)
    observables = build_observables(args.obs_type, args.L, args.meas_axis)

    # Backend selection
    if use_aer:
        backend = AerSimulator(
            method="matrix_product_state",
            matrix_product_state_truncation_threshold=args.mps_truncation_threshold,
            matrix_product_state_max_bond_dimension=args.mps_max_bond_dimension,
            max_memory_mb=-1,
        )
        print(
            f"[Backend] Local AerSimulator (MPS) – backend={backend}, truncation_threshold={args.mps_truncation_threshold}"
        )
    else:
        backend = QiskitRuntimeService().backend(args.backend)
        print(f"[Backend] {backend}")

    # Error map (only meaningful for real hardware)
    if not use_aer:
        fig = plot_error_map(backend)
        figure_save(fig_dir / "error_map.pdf", fig=fig)

    # Layout: for Aer just use a straight line; for hardware use EPLG assignment
    if use_aer:
        eplg_qlist = list(range(args.L))
    else:
        eplg_qlist = get_qubit_assignment_from_EPLG(backend, args.L)

    pm_final = generate_preset_pass_manager(
        optimization_level=args.opt_level,
        backend=backend,
        seed_transpiler=args.seed_transpiler,
        initial_layout=eplg_qlist,
    )

    # Transpile final circuit to extract layout
    print("[Transpile] Final circuit…")
    final_isa_circuit = pm_final.run(circuits[-1])
    layout = final_isa_circuit.layout

    # Figures for final circuit
    final_isa_circuit.draw("mpl", idle_wires=False, fold=-1)
    figure_save(fig_dir / "transpiled_all_trotter_steps.pdf")

    final_cx_depth = final_isa_circuit.depth(filter_function=lambda instr: len(instr.qubits) > 1)
    print(f"[Transpile] Final CX-depth: {final_cx_depth}")

    if not use_aer:
        plot_circuit_layout(final_isa_circuit, backend, view="virtual")
        figure_save(fig_dir / "layout_virtual.pdf")
        plot_circuit_layout(final_isa_circuit, backend, view="physical")
        figure_save(fig_dir / "layout_physical.pdf")

    # Apply layout to observables
    isa_observables = [op.apply_layout(layout) for op in observables]

    # Transpile rest with locked layout
    pm_rest = generate_preset_pass_manager(
        optimization_level=args.opt_level,
        backend=backend,
        seed_transpiler=args.seed_transpiler,
        initial_layout=layout.final_index_layout(),
    )
    print("[Transpile] Time-evolution circuits...")
    te_isa = pm_rest.run(circuits[:-1], num_processes=args.num_processes)
    isa_circuits = te_isa + [final_isa_circuit]

    # Preserve circuit names after transpilation
    for i, c in enumerate(isa_circuits):
        if i < len(circuits):
            c.name = circuits[i].name

    # Annotate circuit info
    for i, c in enumerate(isa_circuits):
        circuit_info[i]["cx_depth"] = c.depth(filter_function=lambda instr: len(instr.qubits) > 1)
        circuit_info[i]["size"] = c.size()
        circuit_info[i]["num_qubits"] = c.num_qubits

    # Metadata
    experiment_info = {
        "mode": "hardware" if not use_aer else "aer_mps",
        "L": args.L,
        "total_time": args.total_time,
        "slices": args.slices,
        "fixed_dt": args.fixed_dt,
        "num_trotter_steps": args.num_trotter_steps,
        "final_cx_depth": final_cx_depth,
        "Jx": args.Jx,
        "Jy": args.Jy,
        "Jz": args.Jz,
        "hx": args.hx,
        "hy": args.hy,
        "hz": args.hz,
        "staggered": args.staggered,
        "break_symmetries": args.break_symmetries,
        "product_state": args.product_state,
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
            "antisymmetric": antisymmetric,
        },
        "observable_type": args.obs_type,
        "aer_mps": {
            "mps_truncation_threshold": args.mps_truncation_threshold,
            "mps_max_bond_dimension": args.mps_max_bond_dimension,
        },
        "meas_axis": args.meas_axis,
        "backend": args.backend,
        "shots": args.shots,
        "opt_level": args.opt_level,
        "seed_transpiler": args.seed_transpiler,
        "zne_enabled": bool(args.zne),
        "pea_enabled": bool(args.pea),
        "dd_enabled": bool(args.dd),
        "pt_enabled": bool(args.pt),
        "git_commit": get_git_commit(),
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp": dt.datetime.now(tz=dt.UTC).isoformat(timespec="seconds"),
    }
    atomic_json_dump(experiment_info, result_dir / "experiment_information.json")

    if use_aer:
        # Local execution with Aer EstimatorV2
        print("[Run] Executing locally on Aer MPS backend…")

        estimator = AerEstimatorV2.from_backend(
            backend,
            options={
                "run_options": {"shots": args.shots},
            },
        )

        # Make a "results" subdir, mirroring the hardware collection script
        results_dir = result_dir
        results_dir.mkdir(parents=True, exist_ok=True)

        jobs = []
        job_meta = []

        # Start timing for Aer MPS execution
        time_start_aer = time.perf_counter()
        circuit_times = []

        for i, c in enumerate(isa_circuits):
            # Time individual circuit execution
            time_start_circuit = time.perf_counter()

            # Run locally
            job = estimator.run([(c, isa_observables)])
            # Aer JobV2 still has a job_id, but we can also tag it explicitly
            try:
                jid = job.job_id()
            except AttributeError, RuntimeError:
                jid = f"local-aer-{i}"

            print(f"  Local job ID: {jid}  Index: {i}  Name: {c.name}")
            jobs.append(job)

            # Get the PrimitiveResult immediately
            try:
                result = job.result()

                time_end_circuit = time.perf_counter()
                circuit_time = time_end_circuit - time_start_circuit
                circuit_times.append(circuit_time)

                # Match the structure from collect_quench.py
                result_dict = {
                    "job_id": jid,
                    "circuit_name": c.name,
                    "circuit_information": circuit_info[i],
                    "PrimitiveResult.metadata": result.metadata,
                    "PubResult.metadata": result[0].metadata,
                    "PubResult.data": result[0].data.__dict__,
                    "execution_time_seconds": circuit_time,
                }

                result_file = results_dir / f"{c.name}_{jid}_results.h5"
                save_result(str(result_file), result_dict)

                print(f"    Saved local results to {result_file} (time: {circuit_time:.3f}s)")
            except (OSError, KeyError, AttributeError) as e:
                print(f"    Error saving local results for job {jid}: {e}")
                circuit_times.append(0.0)

            job_meta.append(
                {
                    "job_id": jid,
                    "circuit_name": c.name,
                    "circuit_information": circuit_info[i],
                    "execution_time_seconds": circuit_times[-1],
                }
            )

        time_end_aer = time.perf_counter()
        total_aer_time = time_end_aer - time_start_aer

        print(f"\n[Timing] Total Aer MPS execution time: {total_aer_time:.3f} seconds")
        print(f"[Timing] Average time per circuit: {np.mean(circuit_times):.3f} seconds")

        # Add timing info to experiment_info
        experiment_info["timing"] = {
            "total_execution_time_seconds": total_aer_time,
            "circuit_execution_times_seconds": circuit_times,
            "average_circuit_time_seconds": float(np.mean(circuit_times)),
            "min_circuit_time_seconds": float(np.min(circuit_times)),
            "max_circuit_time_seconds": float(np.max(circuit_times)),
        }

        # Still write submitted_jobs.json so downstream scripts can inspect metadata
        atomic_json_dump(job_meta, result_dir / "submitted_jobs.json")

        # After aer_mps execution completes, automatically plot results
        print("\n[Plot] Generating plots for aer_mps results...")
        try:
            plot_script = Path(__file__).parent.parent / "scripts" / "plot_results.py"
            subprocess.run([sys.executable, str(plot_script), str(result_dir.parent)], check=True)
            print(f"[Plot] Successfully generated plots in {fig_dir}")
        except (OSError, subprocess.CalledProcessError) as e:
            print(f"[Plot] Warning: Failed to generate plots: {e}")

    else:
        with Batch(backend=backend) as batch:
            print("[Run] Submitting jobs…")
            options = EstimatorOptions(default_shots=args.shots, resilience_level=0)
            # Readout mitigation on
            resilience: ResilienceOptionsV2 = options.resilience  # ty: ignore[invalid-assignment]
            twirling: TwirlingOptions = options.twirling  # ty: ignore[invalid-assignment]
            dd: DynamicalDecouplingOptions = options.dynamical_decoupling  # ty: ignore[invalid-assignment]
            resilience.measure_mitigation = True
            resilience.measure_noise_learning.num_randomizations = 500  # ty: ignore[invalid-assignment]  # default 32
            resilience.measure_noise_learning.shots_per_randomization = "auto"  # ty: ignore[invalid-assignment]
            # Optional toggles
            if args.pt:
                twirling.enable_measure = True
                twirling.enable_gates = True
                twirling.num_randomizations = 500  # default 32
                twirling.shots_per_randomization = "auto"
                print("[QEM] Enabling Pauli Twirling", twirling)
            if args.dd:
                dd.enable = True
                dd.sequence_type = "XpXm"
                print("[QEM] Enabling Dynamical Decoupling", dd)
            if args.zne:
                resilience.zne_mitigation = True
                resilience.zne.noise_factors = (1.0, 1.5, 2.0, 2.5, 3.0)  # ty: ignore[invalid-assignment]
                resilience.zne.extrapolator = (  # ty: ignore[invalid-assignment]
                    "linear",
                    "polynomial_degree_2",
                    "exponential",
                )  # "exponential",
                print("[QEM] Enabling ZNE", resilience.zne)
            if args.pea:
                resilience.zne.amplifier = "pea"  # ty: ignore[invalid-assignment]
                resilience.layer_noise_learning.max_layers_to_learn = 4  # ty: ignore[invalid-assignment]
                resilience.layer_noise_learning.num_randomizations = 32  # ty: ignore[invalid-assignment]
                resilience.layer_noise_learning.shots_per_randomization = 128  # ty: ignore[invalid-assignment]
                resilience.layer_noise_learning.layer_pair_depths = (  # ty: ignore[invalid-assignment]
                    0,
                    1,
                    2,
                    4,
                    16,
                    32,
                )
                print("[QEM] Enabling PEA", resilience.layer_noise_learning)
            estimator = RuntimeEstimatorV2(mode=batch, options=options)

            jobs = []
            job_meta = []
            for i, c in enumerate(isa_circuits):
                job = estimator.run([(c, isa_observables)])
                jid = job.job_id()
                print(f"  Job ID: {jid}  Index: {i}  Name: {c.name}")
                jobs.append(job)
                job_meta.append(
                    {
                        "job_id": jid,
                        "circuit_name": c.name,
                        "circuit_information": circuit_info[i],
                    }
                )

            atomic_json_dump(job_meta, result_dir / "submitted_jobs.json")
