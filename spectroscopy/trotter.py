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

import numpy as np
from qiskit import QuantumCircuit


def xyz_trotter_circuit(
    qc: QuantumCircuit,
    num_steps: int,
    dt: float,
    *,
    Jx: float = 1.0,
    Jy: float = 1.0,
    Jz: float = 1.0,
    hx: float = 0.0,
    hy: float = 0.0,
    hz: float = 0.0,
    staggered: bool = False,
    second_order: bool = True,
    periodic: bool = False,
) -> QuantumCircuit:
    """
    Evolve a QuantumCircuit under the Trotterised XYZ Hamiltonian (same parameterisation as TeNPy's
    SpinModel):

    H = Σ_i (Jx σ_x_i ⛒ σ_x_{i+1} + Jy σ_y_i ⛒ σ_y_{i+1} + Jz σ_z_i ⛒ σ_z_{i+1})
        - Σ_i (hx σ_x_i + hy σ_y_i + hz σ_z_i)
      = 1/4 Σ_i (Jx X_i ⛒ X_{i+1} + Jy Y_i ⛒ Y_{i+1} + Jz Z_i ⛒ Z_{i+1})
        - 1/2 Σ_i (hx X_i + hy Y_i + hz Z_i)

    Where σ_x = X / 2 is a spin operator, and X is the Pauli operator (Analogous for Y, Z).

    Args:
        qc: A QuantumCircuit representing the input state (t = t_0).
        num_steps: The number of Trotter steps to append.
        dt: The Trotter step size.
        Jx, Jy, Jz: Hamiltonian parameters.
        hx, hy, hz: External field Hamiltonian parameters.
        staggered: If True, apply staggered field pattern (alternating +h/-h on even/odd sites).
        second_order: Set to True for 2nd order Trotterisation, False for 1st order.
        periodic: Set to True for periodic boundary conditions, False for open boundary conditions.

    Returns:
        A QuantumCircuit representing the evolved state (t = t_0 + num_ts * dt).
    """
    n = qc.num_qubits
    qc = qc.copy()

    def trotter_block(k: int, params):
        """
        Circuit representing e^i(alpha*XX + beta*YY + gamma*ZZ), where params = [alpha, beta, gamma]
        """
        theta = np.pi / 2 - 2 * params[2]
        phi = 2 * params[0] - np.pi / 2
        lamb = np.pi / 2 - 2 * params[1]

        qc.rz(-np.pi / 2, (k + 1) % n)
        qc.cx((k + 1) % n, k)
        qc.rz(theta, k)
        qc.ry(phi, (k + 1) % n)
        qc.cx(k, (k + 1) % n)
        qc.ry(lamb, (k + 1) % n)
        qc.cx((k + 1) % n, k)
        qc.rz(np.pi / 2, k)

    # Convert Jx, Jy, Jy to alpha, beta, gamma for trotter_block
    parameters = np.array([-Jx * dt / 4, -Jy * dt / 4, -Jz * dt / 4])

    # Build the main part of the 1st or 2nd order Trotter circuit.
    for j in range(num_steps):
        # 1st half of a layer: even terms
        for q in range(0, n - 1, 2):
            trotter_block(q, parameters / 2 if second_order and j == 0 else parameters)

        # Add PBC term for even layer if n is even
        if periodic and n % 2 == 0:
            trotter_block(n - 1, parameters / 2 if second_order and j == 0 else parameters)

        # External fields (hx, hy, hz) - applied after even layer
        for q in range(n):
            sign = (-1) ** q if staggered else 1
            if hx != 0:
                qc.rx(-sign * hx * dt / 2 if second_order else -sign * hx * dt, q)
            if hy != 0:
                qc.ry(-sign * hy * dt / 2 if second_order else -sign * hy * dt, q)
            if hz != 0:
                qc.rz(-sign * hz * dt / 2 if second_order else -sign * hz * dt, q)

        # 2nd half of a layer: odd terms
        for q in range(1, n - 1, 2):
            trotter_block(q, parameters)

        # Add PBC term for odd layer if n is odd
        if periodic and n % 2 == 1:
            trotter_block(n - 1, parameters)

        # External fields (hx, hy, hz) - applied after odd layer (second_order only)
        if second_order:
            for q in range(n):
                sign = (-1) ** q if staggered else 1
                if hx != 0:
                    qc.rx(-sign * hx * dt / 2, q)
                if hy != 0:
                    qc.ry(-sign * hy * dt / 2, q)
                if hz != 0:
                    qc.rz(-sign * hz * dt / 2, q)

    # For 2nd order Trotter, we add an extra half-layer identical to the front one.
    if second_order:
        for q in range(0, n - 1, 2):
            trotter_block(q, parameters / 2)
        if periodic and n % 2 == 0:
            trotter_block(n - 1, parameters / 2)

    return qc
