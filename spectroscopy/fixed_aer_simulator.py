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

# This code is based on and overwrites the _set_method_config method from Qiskit AerSimulator.
# This is done to increase the maximum number of qubits on the MPS backend from 63 to 156.

from qiskit_aer.backends.aer_simulator import AerSimulator
from qiskit_aer.backends.backend_utils import (
    MAX_QUBITS_STATEVECTOR,
)


class FixedAerSimulator(AerSimulator):
    def __init__(
        self,
        configuration=None,
        properties=None,
        provider=None,
        target=None,
        **backend_options,
    ):
        super().__init__(configuration, properties, provider, target, **backend_options)

    def _set_method_config(self, method=None):
        """Set non-basis gate options when setting method"""
        # Update configuration description and number of qubits
        if method == "statevector":
            description = "A C++ statevector simulator with noise"
            n_qubits = MAX_QUBITS_STATEVECTOR
        elif method == "density_matrix":
            description = "A C++ density matrix simulator with noise"
            n_qubits = MAX_QUBITS_STATEVECTOR // 2
        elif method == "unitary":
            description = "A C++ unitary matrix simulator"
            n_qubits = MAX_QUBITS_STATEVECTOR // 2
        elif method == "superop":
            description = "A C++ superop matrix simulator with noise"
            n_qubits = MAX_QUBITS_STATEVECTOR // 4
        elif method == "matrix_product_state":
            description = "A C++ matrix product state simulator with noise"
            n_qubits = 156  # TODO: not sure what to put here?
        elif method == "stabilizer":
            description = "A C++ Clifford stabilizer simulator with noise"
            n_qubits = 10000  # TODO: estimate from memory
        elif method == "extended_stabilizer":
            description = "A C++ Clifford+T extended stabilizer simulator with noise"
            n_qubits = 63  # TODO: estimate from memory
        else:
            # Clear options to default
            description = None
            n_qubits = None

        if self._configuration.coupling_map:
            n_qubits = max(list(map(max, self._configuration.coupling_map))) + 1

        self._set_configuration_option("description", description)
        self._set_configuration_option("n_qubits", n_qubits)
