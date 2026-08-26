# quench-spectroscopy-qpu

> [!NOTE]
    > No updates should be expected to this repository. This is not officially supported IBM Quantum software.

Code to reproduce all results and figures from:

    D. Millar et al., "Quench Spectroscopy of Magnetic Excitations on a Superconducting Quantum Processor",
    arXiv:2607.02673 (2026). https://arxiv.org/abs/2607.02673

If you use this code or data in your research, please cite:

```bib
@article{millar2026quench,
  title={Quench Spectroscopy of Magnetic Excitations on a Superconducting Quantum Processor},
  author={Millar, DA and Pennington, GW and Siow, NTM and Brandhofer, S and Crain, J and Essler, FHL and Green, AG and Thomson, SJ},
  journal={arXiv preprint arXiv:2607.02673},
  year={2026}
}
```

## License

The source code is licensed under the [Apache License 2.0](LICENSE.txt).

The results data (contents of `experiments/`) are licensed under the
[Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/).
If you use the data, please cite the paper above.

## Quick start

```bash
# Install core dependencies (plotting and simulation)
uv sync
source .venv/bin/activate

# Reproduce all paper figures
python scripts/plot_paper_results.py
```

> Scripts can also be run without activating the environment using `uv run`:
> ```bash
> uv run python scripts/plot_paper_results.py
> ```

## Running new experiments

`scripts/run_quench.py` requires additional dependencies for MPS-to-circuit
conversion and AQC-tensor simulation. Install them with the `run` extra:

```bash
uv sync --extra run

# Run a TenPy simulation
python scripts/run_quench.py \
    --mode tenpy -L 51 --Jx -1.0 --Jy -1.0 --Jz 2.0 \
    --total-time 10 --slices 51 --num-trotter-steps 5 \
    --meas-axis sy --obs-type pauli --plot

# Submit to IBM Quantum hardware
python scripts/run_quench.py \
    --mode hardware -L 51 --Jx -1.0 --Jy -1.0 --Jz 2.0 \
    --backend ibm_boston --shots 10000

# Collect completed jobs
python scripts/collect_quench.py experiments/<experiment_dir>

# Plot results
python scripts/plot_results.py experiments/<experiment_dir>
```

## Hardware access

The scripts expect your `QiskitRuntimeService` credentials to be saved locally. Retrieve these from <https://quantum.cloud.ibm.com>.

You can save these using the following snippet:

```python
from qiskit_ibm_runtime import QiskitRuntimeService

QiskitRuntimeService.save_account(
    token="YOUR-TOKEN",
    instance="YOUR-INSTANCE",
    overwrite=True,
    set_as_default=True,
)
```

## Linting, formatting and type checking

```bash
# Install dev dependencies
uv sync --extra dev

uv run ruff check
uv run ruff format
uv run ty check
```

## Structure

```
spectroscopy/               # Core library
    fourier.py              # 2D Fourier transform utilities
    utils.py                # General utility functions
    evolution.py            # Build Trotter evolution circuits
    trotter.py              # Trotter circuits
    hardware_utils.py       # Hardware helpers (observables, qubit layout)
    run_hardware.py         # Run experiment on IBM Quantum hardware or Aer MPS
    run_tenpy.py            # Run experiment with TenPy TDVP
    fixed_aer_simulator.py  # Patched AerSimulator with extended MPS qubit limit
    plotting.py             # Results plotting and Bethe-ansatz spectrum overlays

scripts/
    plot_paper_results.py   # Reproduce all figures from arXiv:2607.02673
    run_quench.py           # Prepare ground state and submit circuits
    collect_quench.py       # Collect completed IBM Quantum jobs
    plot_results.py         # Plot dynamics + QSF from saved results

experiments/                # Experiment output with results from arXiv:2607.02673

figures/                    # Output figures (gitignored)
```

## Acknowledgements

This work was supported by the Hartree National Centre for Digital Innovation, a collaboration
between the Science and Technology Facilities Council and IBM.
