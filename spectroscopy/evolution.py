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


from qiskit import QuantumCircuit

from spectroscopy.trotter import xyz_trotter_circuit


def build_xyz_evolution_circuits(
    quench_circuit: QuantumCircuit,
    total_time: float,
    slices: int,
    fixed_dt: bool,
    num_trotter_steps: int,
    Jx: float,
    Jy: float,
    Jz: float,
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    staggered: bool = False,
) -> tuple[list[QuantumCircuit], list[dict]]:
    """
    Builds circuits corresponding to evolution under H following a quench of the ground state.

    This builds one circuit per time-slice to be measured for spectroscopy.

    Args:
        quench_circuit: The circuit representing the ground state followed by the local quench.
        total_time: The longest time to evolve to.
        slices: The number of time-slices to measure.
        fixed_dt: If True, fix dt and evolve by adding one Trotter step per slice. If False,
            fix the number of Trotter steps to num_trotter_steps and evolve by varying dt.
        num_trotter_steps: The fixed number of Trotter steps to include in each circuit, ONLY
            if fixed_dt==False.
        Jx: The coupling strength.
        Jy: The coupling strength.
        Jz: The coupling strength.
        hx: The external field strength along x-axis.
        hy: The external field strength along y-axis.
        hz: The external field strength along z-axis.
        staggered: If True, apply staggered field pattern (alternating +h/-h on even/odd sites).

    Returns:
        The circuits: the number of circuits is equal to `slices`
        circuit_information: information about each circuit.
    """

    if fixed_dt:
        dt = total_time / (slices - 1)

    circuits = []
    circuit_information = []

    for slice in range(slices):
        time = slice / (slices - 1) * total_time
        if slice > 0:
            evolved_qc = xyz_trotter_circuit(
                qc=quench_circuit,
                num_steps=slice if fixed_dt else num_trotter_steps,
                dt=dt if fixed_dt else time / num_trotter_steps,
                Jx=Jx,
                Jy=Jy,
                Jz=Jz,
                hx=hx,
                hy=hy,
                hz=hz,
                staggered=staggered,
                second_order=True,
            )
        else:
            evolved_qc = quench_circuit.copy()

        evolved_qc.name = f"evol_t{time:.3f}_idx{slice}"
        circuits.append(evolved_qc)
        circuit_information.append(
            {
                "time": time,
                "num_trotter_steps": slice if fixed_dt else num_trotter_steps,
                "dt": dt if fixed_dt else time / num_trotter_steps,
            }
        )
    return circuits, circuit_information
