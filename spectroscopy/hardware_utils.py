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

import json
import os

import h5py
import numpy as np
from qiskit.quantum_info import SparsePauliOp


def average_sides(arr: np.ndarray) -> np.ndarray:
    """Symmetrically averages an array around its center.

    Args:
        arr: The input array to be symmetrically averaged.

    Returns:
        The symmetrically averaged array.
    """
    arr = np.asarray(arr)
    return (arr + arr[::-1]) / 2


def save_result(path: str, result_dict: dict) -> None:
    """Save a per-circuit result dictionary to an HDF5 file.

    Scalar/string fields are stored as dataset attributes on the root group.
    Numpy arrays are stored as top-level datasets.
    Nested dicts (circuit_information, PrimitiveResult.metadata,
    PubResult.metadata) are stored as JSON-encoded string attributes.
    The PubResult.data dict may contain numpy arrays — each entry is stored
    as its own dataset named ``data/<key>``.

    Args:
        path: Destination file path (should end in ``.h5``).
        result_dict: Dictionary as assembled in run_hardware / collect_quench.
    """
    with h5py.File(path, "w") as fh:
        # scalar / string top-level fields
        for key in ("job_id", "circuit_name"):
            fh.attrs[key] = result_dict[key]

        # timing fields (optional, may be None)
        for key in ("execution_time_seconds", "qpu_execution_time_seconds"):
            if key in result_dict and result_dict[key] is not None:
                fh.attrs[key] = result_dict[key]

        # nested metadata dicts → JSON attributes
        for key in ("circuit_information", "PrimitiveResult.metadata", "PubResult.metadata"):
            fh.attrs[key] = json.dumps(result_dict[key])

        # PubResult.data: mix of arrays and scalars
        data_grp = fh.create_group("data")
        for k, v in result_dict["PubResult.data"].items():
            if isinstance(v, np.ndarray):
                data_grp.create_dataset(k, data=v)
            elif v is not None:
                data_grp.attrs[k] = v


def load_result(path: str) -> dict:
    """Load a per-circuit result from an HDF5 file written by :func:`save_result`.

    Returns a dict with the same structure used by the rest of the codebase:
    ``result_dict["PubResult.data"]`` is a plain dict of numpy arrays / scalars,
    and metadata fields are deserialized from JSON.

    Args:
        path: Path to the ``.h5`` file.

    Returns:
        Dictionary matching the structure saved by :func:`save_result`.
    """
    result_dict: dict = {}
    with h5py.File(path, "r") as fh:
        for key in ("job_id", "circuit_name"):
            if key in fh.attrs:
                result_dict[key] = str(fh.attrs[key])

        for key in ("execution_time_seconds", "qpu_execution_time_seconds"):
            if key in fh.attrs:
                result_dict[key] = float(fh.attrs[key])

        for key in ("circuit_information", "PrimitiveResult.metadata", "PubResult.metadata"):
            if key in fh.attrs:
                result_dict[key] = json.loads(fh.attrs[key])

        pub_data: dict = {}
        if "data" in fh:
            data_grp = fh["data"]
            for k in data_grp:
                pub_data[k] = data_grp[k][()]
            for k, v in data_grp.attrs.items():
                pub_data[k] = v
        result_dict["PubResult.data"] = pub_data

    return result_dict


def process_observations(obs_list: np.ndarray) -> np.ndarray:
    """
    # Average the observations either side of the observation.
    """
    return np.array([average_sides(obs) for obs in obs_list])


def read_observation_data(
    experiment_directory: str,
    time_resolution: float,
    slices: int,
    L: int,
    raw: bool = True,
) -> np.ndarray:
    """
    Read and organize observation data.
    """
    results_dir = os.path.join(experiment_directory, "results")

    with open(os.path.join(results_dir, "experiment_information.json"), "r") as fh:
        info = json.load(fh)
        obs_type = info["observable_type"]

    if obs_type is not None:
        s = str(obs_type)
        if "3" in s:
            sites_per_obs = 3
        elif "2" in s or obs_type == "Dx":
            sites_per_obs = 2
        else:
            sites_per_obs = 1

    expected_len = max(0, L - sites_per_obs + 1)
    obs_list = np.zeros((slices, expected_len))
    for filename in os.listdir(results_dir):
        if not filename.endswith(".h5"):
            continue
        result_dict = load_result(os.path.join(results_dir, filename))
        time = result_dict["circuit_information"]["time"]
        index = round(time / time_resolution)

        if raw:
            #  Use ZNE unmitigated data.
            obs_list[index] = result_dict["PubResult.data"]["evs_noise_factors"][:, 0]
        else:
            #  Use ZNE extrapolated data.
            obs_list[index] = result_dict["PubResult.data"]["evs"]
    return obs_list


def _score(backend, qargs):
    """
    Get an 'average' error that does not consider the exact number
    and type of two-qubit gates in the circuit
    """
    target = backend.target
    qarg_error = 0
    for op in target.operation_names_for_qargs(qargs):
        inst_props = target[op].get(qargs, None)
        if inst_props is not None and inst_props.error is not None:
            qarg_error += inst_props.error

    for q in qargs:
        inst_props = target["measure"].get(q, None)
        if inst_props is not None and inst_props.error is not None:
            qarg_error += inst_props.error
    print("qarg error", qarg_error)
    return qarg_error


def extend_eplg_list_by_one(backend, qlist):
    """
    Tries to extend the given list of qubits by one qubit with the lowest average error.
    """
    qset = set(qlist)

    try:
        pred = next(s for s in set(backend.coupling_map.neighbors(qlist[0])).difference(qset))
        pred_score = _score(backend, (pred, qlist[0]))
    except (StopIteration, ValueError) as e:
        print(e)
        pred_score = 1.0

    try:
        succ = next(s for s in set(backend.coupling_map.neighbors(qlist[-1])).difference(qset))
        succ_score = _score(backend, (qlist[-1], succ))
    except (StopIteration, ValueError) as e:
        print(e)
        succ_score = 1.0

    if pred_score > 0.1 and succ_score > 0.1:
        # wouldn't advise to extend the qlist
        print("Tried to extend the EPLG chain but found only bad qubits")
        raise RuntimeError("Tried to extend the EPLG chain but found only bad qubits")

    if pred_score < succ_score:
        return [pred] + qlist
    else:
        return qlist + [succ]


def get_qubit_assignment_from_EPLG(backend, num_qubits):
    """
    Get an initial qubit assignment from the EPLG metric.
    The output is a list of integer q that maps qubit i
    in the circuit to qubit q[i] on the device.
    """
    props = backend.properties()
    EPLG = props.general_qlists

    for qlist_dict in EPLG:
        qlist = qlist_dict["qubits"]
        if len(qlist) == num_qubits:
            return qlist
    assert num_qubits > 100
    qlist = EPLG[-1]["qubits"]
    for _ in range(num_qubits - 100):
        qlist = extend_eplg_list_by_one(backend, qlist)
    return qlist


def _pauli_string(L: int, paulis: list[tuple[int, str]]) -> str:
    """Return Pauli string (Qiskit ordering: rightmost qubit is index 0)."""
    s = ["I"] * L
    for idx, ch in paulis:
        s[L - 1 - idx] = ch
    return "".join(s)


def _axis_to_char(axis: str) -> str:
    return {"sx": "X", "sy": "Y", "sz": "Z"}[axis.lower()]


def build_observables(obs_type: str, L: int, meas_axis: str) -> list[SparsePauliOp]:
    """Construct SparsePauliOp observables for given observable type."""
    if obs_type == "pauli":
        # per-site single-axis Pauli, e.g. Z, X, or Y
        return [SparsePauliOp(_pauli_string(L, [(i, _axis_to_char(meas_axis))])) for i in range(L)]

    if obs_type == "pauli2":
        # bond-centered adjacent correlator <sigma_a(i) sigma_a(i+1)>
        # Returns list of length L-1
        ch = _axis_to_char(meas_axis)
        return [SparsePauliOp(_pauli_string(L, [(j, ch), (j + 1, ch)])) for j in range(L - 1)]

    if obs_type == "Dx":
        # bond-centered adjacent correlator <sigma_a(i) sigma_a(i+1)>
        # Returns list of length L-1
        ch = _axis_to_char(meas_axis)
        return [SparsePauliOp(_pauli_string(L, [(j, ch), (j + 1, ch)])) for j in range(L - 1)]

    if obs_type == "flip1":
        # flip1 ≡ S+ + S− ∝ σx
        return [SparsePauliOp(_pauli_string(L, [(i, "X")])) for i in range(L)]

    if obs_type == "flip2":
        # flip2 ≡ S+S+ + S−S− ∝ XX − YY
        obs = []
        for j in range(L - 1):
            xx = SparsePauliOp(_pauli_string(L, [(j, "X"), (j + 1, "X")]))
            yy = SparsePauliOp(_pauli_string(L, [(j, "Y"), (j + 1, "Y")]))
            obs.append(xx - yy)
        return obs

    if obs_type == "flip2":
        # ≡ XX + YY + ZZ
        obs = []
        for j in range(L - 1):
            xx = SparsePauliOp(_pauli_string(L, [(j, "X"), (j + 1, "X")]))
            yy = SparsePauliOp(_pauli_string(L, [(j, "Y"), (j + 1, "Y")]))
            obs.append(xx - yy)
        return obs

    if obs_type == "projector2":
        # P↑↑(j,j+1) = (I + Zj + Z_{j+1} + ZjZ_{j+1})/4
        obs = []
        id = SparsePauliOp("I" * L)
        for j in range(L - 1):
            zj = SparsePauliOp(_pauli_string(L, [(j, "Z")]))
            zj1 = SparsePauliOp(_pauli_string(L, [(j + 1, "Z")]))
            zz = SparsePauliOp(_pauli_string(L, [(j, "Z"), (j + 1, "Z")]))
            obs.append(0.25 * (id + zj + zj1 + zz))
        return obs

    if obs_type == "projector3":
        # P↑↑↑(j-1,j,j+1) = (I + Zj-1 + Zj + Zj+1 + Zj-1Zj + ZjZj+1 + Zj-1Zj+1 + Zj-1ZjZj+1)/8
        # Centered on site j, spanning j-1, j, j+1
        obs = []
        id = SparsePauliOp("I" * L)
        for j in range(1, L - 1):  # j ranges from 1 to L-2 (sites j-1, j, j+1 exist)
            z_jm1 = SparsePauliOp(_pauli_string(L, [(j - 1, "Z")]))
            z_j = SparsePauliOp(_pauli_string(L, [(j, "Z")]))
            z_jp1 = SparsePauliOp(_pauli_string(L, [(j + 1, "Z")]))
            zz_jm1_j = SparsePauliOp(_pauli_string(L, [(j - 1, "Z"), (j, "Z")]))
            zz_j_jp1 = SparsePauliOp(_pauli_string(L, [(j, "Z"), (j + 1, "Z")]))
            zz_jm1_jp1 = SparsePauliOp(_pauli_string(L, [(j - 1, "Z"), (j + 1, "Z")]))
            zzz = SparsePauliOp(_pauli_string(L, [(j - 1, "Z"), (j, "Z"), (j + 1, "Z")]))
            obs.append(0.125 * (id + z_jm1 + z_j + z_jp1 + zz_jm1_j + zz_j_jp1 + zz_jm1_jp1 + zzz))
        return obs

    if obs_type == "Dx":
        # Domain-wall (kink) density on each bond: D_i = (1 + Z_i Z_{i+1})/2
        obs = []
        id = SparsePauliOp("I" * L)
        for j in range(L - 1):
            zz = SparsePauliOp(_pauli_string(L, [(j, "Z"), (j + 1, "Z")]))
            obs.append(0.5 * (id + zz))
        return obs

    raise ValueError(f"Unsupported obs_type for hardware: {obs_type!r}")
