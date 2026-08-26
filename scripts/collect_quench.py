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
Script to collect multiple jobs from a batch:
"""

import json
import os
import sys

import numpy as np
from qiskit_ibm_runtime import (
    IBMRuntimeError,
    QiskitRuntimeService,
)

from spectroscopy.hardware_utils import load_result, save_result


def main():
    if len(sys.argv) != 2:
        print("Usage: python collect_quench.py <experiment_directory>")
        sys.exit(1)

    experiment_directory = sys.argv[1]
    print(f"Collecting data for {experiment_directory}")

    collect_results(experiment_directory)


def collect_results(result_dir):
    result_dir = f"{result_dir}/results/"

    # Ensure the directory exists
    os.makedirs(result_dir, exist_ok=True)

    service = QiskitRuntimeService()

    # Load job metadata
    with open(f"{result_dir}submitted_jobs.json", "r") as f:
        job_metadata = json.load(f)

    completed_jobs = []

    for entry in job_metadata:
        job_id = entry["job_id"]
        circuit_information = entry["circuit_information"]

        try:
            job = service.job(job_id)
            if job.done():
                result = job.result()
                result_file = os.path.join(
                    result_dir, f"{entry['circuit_name']}_{job_id}_results.h5"
                )

                # Save result
                try:
                    # Extract QPU execution time from job metrics
                    execution_time = None
                    try:
                        job_metrics = job.metrics()
                        if job_metrics and "usage" in job_metrics:
                            execution_time = job_metrics["usage"].get("seconds")

                        if execution_time is not None:
                            print(f"  QPU execution time: {execution_time} seconds")
                    except (IBMRuntimeError, KeyError, AttributeError) as e:
                        print(f"  Note: Could not extract execution time from job metrics: {e}")

                    result_dict = {
                        "job_id": job_id,
                        "circuit_name": entry["circuit_name"],
                        "circuit_information": circuit_information,
                        "PrimitiveResult.metadata": result.metadata,
                        "PubResult.metadata": result[0].metadata,
                        "PubResult.data": result[0].data.__dict__,
                        "qpu_execution_time_seconds": execution_time,
                    }

                    save_result(result_file, result_dict)

                    completed_jobs.append(job_id)
                    print(f"Successfully retrieved results for job {job_id}.")
                except (IBMRuntimeError, OSError, KeyError, AttributeError) as e:
                    print(f"Error saving results for job {job_id}: {e!s}")
            else:
                print(f"Job {job_id} is not DONE. Status: {job.status()!s}")
        except IBMRuntimeError as e:
            print(f"Error retrieving job {job_id}: {e!s}")

    print(f"Retrieved results for {len(completed_jobs)} jobs.")

    # Check if this is an antisymmetric experiment and compute difference
    try:
        exp_info_path = os.path.join(result_dir, "experiment_information.json")
        with open(exp_info_path, "r") as f:
            exp_info = json.load(f)

        if exp_info.get("quench", {}).get("antisymmetric", False):
            print("[Antisymmetric] Computing antisymmetric difference...")
            compute_antisymmetric_difference(result_dir, exp_info)
    except (OSError, KeyError, ValueError) as e:
        print(f"Note: Could not check for antisymmetric mode: {e}")


def compute_antisymmetric_difference(result_dir, exp_info):
    """
    Compute antisymmetric difference from +θ and -θ results.
    Saves: obs_plus_theta.npy, obs_minus_theta.npy, obs_antisymmetric.npy
    """
    L = exp_info["L"]
    slices = exp_info["slices"]
    total_time = exp_info["total_time"]
    time_resolution = total_time / (slices - 1)

    # Determine the number of sites per observable
    obs_type = exp_info.get("observable_type")
    if obs_type is not None:
        s = str(obs_type)
        if "3" in s:
            sites_per_obs = 3
        elif "2" in s or obs_type == "Dx":
            sites_per_obs = 2
        else:
            sites_per_obs = 1
    else:
        sites_per_obs = 1

    # Calculate expected observable array length
    expected_len = max(0, L - sites_per_obs + 1)

    # Separate plus and minus results based on circuit_information metadata
    import glob

    result_files = glob.glob(f"{result_dir}*_results.h5")

    obs_plus = np.zeros((slices, expected_len))
    obs_minus = np.zeros((slices, expected_len))

    plus_count = 0
    minus_count = 0

    # Load all results and separate by quench_sign
    for f in result_files:
        data = load_result(f)
        circuit_info = data["circuit_information"]

        # Check if this result has a quench_sign field
        quench_sign = circuit_info.get("quench_sign")
        if quench_sign is None:
            continue  # Skip non-antisymmetric results

        # Get time and compute index (same logic as read_observation_data)
        time = circuit_info["time"]
        time_idx = round(time / time_resolution)
        evs = data["PubResult.data"]["evs"]

        if quench_sign == "plus":
            obs_plus[time_idx, :] = evs[:expected_len]
            plus_count += 1
        elif quench_sign == "minus":
            obs_minus[time_idx, :] = evs[:expected_len]
            minus_count += 1

    if plus_count == 0 or minus_count == 0:
        print(f"[Antisymmetric] Warning: Found {plus_count} +θ and {minus_count} -θ results")
        print("[Antisymmetric] Need both +θ and -θ results to compute antisymmetric difference")
        return

    print(f"[Antisymmetric] Found {plus_count} +θ and {minus_count} -θ results")

    # Compute antisymmetric difference: ΔA(t) = [A₊ - A₋] / 2
    obs_antisymmetric = (obs_plus - obs_minus) / 2.0

    # Save all three arrays to the results directory
    np.save(os.path.join(result_dir, "obs_plus_theta.npy"), obs_plus)
    np.save(os.path.join(result_dir, "obs_minus_theta.npy"), obs_minus)
    np.save(os.path.join(result_dir, "obs_antisymmetric.npy"), obs_antisymmetric)

    print(
        f"[Antisymmetric] Saved antisymmetric difference to {os.path.join(result_dir, 'obs_antisymmetric.npy')}"
    )
    print("[Antisymmetric] Also saved raw +θ and -θ data for debugging")


if __name__ == "__main__":
    main()
