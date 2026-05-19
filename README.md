# PyRInst

## Installation Guide

### Compatibility Matrix

| Category | Current Recommendation |
| --- | --- |
| Python | 3.11-3.13 |
| Core dependencies | `numpy`, `scipy`, `matplotlib` |
| Optional MACE support | Project-provided `pkgs/mace_torch-*.whl` plus a matching `torch` / CUDA environment |
| Development tools | `pytest`, `ruff`, `build` |
| Documentation tools | `sphinx`, `numpydoc` |

### Basic Installation

If you only need the core functionality, run the following in the project root:

```bash
pip install .
```

After installation, `pyrinst-gen-ref`, `pyrinst-sampling`, `pyrinst-fep-eval`, and `pyrinst-optimize` will be available as command-line programs.

### Optional Dependency Layers

Common installation options:

```bash
pip install ".[dev]"
pip install ".[docs]"
```

What each layer means:

- The default install, `pip install .`, only includes the core numerical dependencies and is suitable for most non-MACE workflows.
- `.[dev]` is intended for local development, testing, and packaging checks.
- `.[docs]` is used to build the Sphinx documentation.
- MACE is not installed automatically through `pyproject.toml`; if you need it, install the project-specified wheel manually.
- The default CI only covers CPU paths; MACE/GPU validation is done through manual scripts.

### Recommended Environment Setup for MACE

The following setup is closer to the environment currently used by this project in practice:

```bash
conda create -n your_name python=3.13
pip install cuequivariance==0.7.0 cuequivariance-torch==0.7.0 cuequivariance-ops-torch-cu12==0.7.0
pip install torch==2.6.0 --index-url https://download.pytorch.org/whl/cu126
pip install pkgs/mace_torch-0.3.14-py3-none-any.whl # This wheel is provided by the project and is based on a modified official mace_torch-0.3.14
```

Things to keep in mind:

- PyRInst does not automatically install the generic `mace-torch` package. If you want to use MACE, install only the wheel provided or specified by this project.
- In the default CI, `test_mace.py` only checks the CPU-side interface. For actual GPU/MACE validation, run these two scripts manually:

```bash
bash tests/gpu/check_mace_env.sh
PYRINST_MACE_MODEL=/abs/path/to/model.model bash tests/gpu/run_mace_release_checks.sh
```

### i-pi `nvt-cc` Bug Fix

For i-pi versions **3.20 and later**, the `nvt-cc` bug has been fixed upstream.  
For older i-pi versions, if upgrading is not possible, apply the following manual patch : In `ipi/engine/motion/dynamics.py`, inside the `step` method of the `NVTCCIntegrator` class, add one extra line below:

```python
# self.qcstep() # for the moment I just avoid doing the centroid step.
self.nm.free_qstep()
```

so that it becomes:

```python
# self.qcstep() # for the moment I just avoid doing the centroid step.
self.nm.free_qstep()
self.nm.free_qstep()
```

Inside a virtual environment, this file is usually located at `path/to/you/venv/lib/pythonX.Y/site-packages/ipi/engine/motion/dynamics.py`, where `path/to/you/venv/` is the root of your conda environment and `pythonX.Y/` should match the Python version you actually installed.

## Workflow

## Harmonic FEP

Harmonic FEP is used to apply a thermodynamic correction, based on a harmonic reference state, to a given centroid configuration at a target temperature, so that a more accurate free-energy estimate can be obtained. The module includes reference-state generation, normal-mode-based sampling, and energy correction calculations caused by differences between potential energy surfaces. To keep the harmonic approximation reasonable, avoid structures with large imaginary frequencies whenever possible.

The overall workflow consists of the following four main steps.

### 1. Generate the Reference Structure

First, compute the Hessian and vibrational frequency information for the centroid geometry and build the reference harmonic oscillator state.

**Related command:** `pyrinst-gen-ref` (entry module: `pyrinst.cli.gen_ref`)

**Example:**

```bash
pyrinst-gen-ref geom.xyz -o ref_out -P MACE --model_path /path/to/model.model --device cuda
```

**Arguments:**

- `input`: Input centroid geometry in XYZ format
- `-o, --output`: Output prefix for the reference-state PKL file (default: `ref`)
- `-P, --PES`: Selected potential energy surface, for example `MACE`
- Parameters specific to the MACE PES:
- `--model_path`: Path to the MACE model
- `--dtype`: Precision, either `float64` or `float32`
- `--device`: Execution device, either `cuda` or `cpu`

After the script finishes, it prints the relevant frequency information and writes a `.pkl` file in the current directory containing key data such as frequencies and normal modes.

---

### 2. Generate Sampled Configurations

Sample the system in normal-mode space under the harmonic potential and generate configurations for each bead.

**Related command:** `pyrinst-sampling` (entry module: `pyrinst.cli.sampling`)

**Example:**

```bash
pyrinst-sampling ref_out.pkl -T 300 -N 4096 -n 24 -o simulation.pos
```

**Arguments:**

- `input`: Path to the `pkl` file generated in step 1
- `-T`: Target sampling temperature in K (default: `300`)
- `-N`: Total number of samples (default: `4096`)
- `-n, --nbeads`: Number of beads (default: `24`)
- `-o, --output`: Prefix for output files (default: `simulation.pos`, which generates files such as `simulation.pos_00.xyz` through `simulation.pos_23.xyz`)
- `--nprandom`: Use NumPy random sampling instead of the default Sobol sampling. This option is useful for high-dimensional systems where SciPy's Sobol sampler exceeds its supported dimensionality limit.

> **Note:** During execution, this script not only writes XYZ configuration files, but also **updates the `pkl` file** with the harmonic reference energy of each bead. Be sure to keep a copy of the updated `pkl` file.

> The results obtained from Sobol sampling and NumPy random sampling may show small differences, but the deviation is typically less than `0.05 meV/atom`.

---

### 3. Evaluate Energies

> This is usually the most time-consuming step. Before running production calculations, plan resource allocation carefully to maximize throughput.

#### Example: Using MACE to Compute Energies

One example script is:

```bash
for d in {00..23}; do
    mace_eval_configs \
    --configs="simulation.pos_${d}.xyz" \
    --model="/path/to/model.model" \
    --output="simulation.pos_eval_${d}.xyz" \
    --device="cuda" \
    --enable_cueq \
    --no_forces \
    --batch_size=1024
done
```

Parameters that matter most for performance:

- `--device`: Use `cuda` whenever possible; CPU evaluation is very slow.
- `--enable_cueq`: This can significantly improve speed while reducing GPU memory usage. It is recommended to build the environment as described above.
- `--no_forces`: This option in `mace_beta` is intended specifically for energy evaluation. It skips force calculations and avoids a large amount of expensive `auto_gradient()` work, which greatly improves speed and also reduces GPU memory usage.
- `--batch_size`: Try to make full use of single-GPU memory, but leave some headroom to avoid occasional `OOM` issues caused by delayed memory release. If an output file is empty or the process stalls on one file, insufficient GPU memory is a likely cause.

---

### 4. Compute the FEP Free-Energy Correction

Use the bead configurations with computed energies together with the `pkl` file to obtain the final free-energy estimate.

**Related command:** `pyrinst-fep-eval` (entry module: `pyrinst.cli.fep_eval`)

**Example:**

```bash
pyrinst-fep-eval ref_out.pkl --prefix simulation.pos_eval -n 24
```

**Arguments:**

- `input`: The `pkl` file generated in step 1
- `--prefix`: Keep this consistent with the prefix used for the evaluated bead files
- `-n, --nbeads`: Number of beads (default: `24`)

**Script output:**

- `reference`: Reference free-energy estimate
- `correction`: Free-energy correction obtained from FEP
- `Delta F`: The **difference** between the centroid free energy at the target temperature and the reference potential energy surface; this is the energy used when training the effective potential
- `uncertainty`: Estimated uncertainty of the correction result, including effects such as autocorrelation

## Instanton FEP

Below the crossover temperature `Tc`, Harmonic FEP can introduce large errors because the harmonic approximation becomes less reliable. Instanton FEP is designed for this low-temperature regime and performs thermodynamic correction estimates based on an instanton reference structure. The overall workflow is similar to Harmonic FEP, but the procedure for finding the reference state is different.

Because this method is built around an instanton-based correction, the workflow consists of the following five main steps.

### 1. Generate the Initial Structure

First, compute the reference harmonic state from the centroid geometry so it can be used as the initial guess for the later optimization.

**Related command:** `pyrinst-gen-ref`

**Example:**

```bash
pyrinst-gen-ref water.xyz -P MACE --model_path MACE-OFF23_medium_water_train3_run-1020_stagetwo.model
```

The default output reference file is `ref.pkl`.

---

### 2. Optimize to the Instanton Structure

After the initial reference structure is generated, perform a fixed-centroid instanton geometry optimization at the target temperature.

**Related command:** `pyrinst-optimize`

**Example:**

```bash
pyrinst-optimize ref.pkl -o inst.pkl -T 300 --mode centroid -P MACE -F MACE-OFF23_medium_water_train3_run-1020_stagetwo.model -N 24 -s 0.189
```

**Arguments:**

- `input`: Path to the `ref.pkl` file generated in step 1
- `-o, --output`: Name of the output PKL file after optimization, for example `inst.pkl`
- `-T`: Target temperature
- `--mode`: Use `centroid` here for fixed-centroid instanton optimization
- `-P, --PES`: Selected potential energy surface
- `-F`: Additional model-path argument corresponding to the selected PES
- `-N`: Number of beads, for example `24`
- `-s, --spread`: Length of the initial instanton guess

> **Tip:** If the instanton optimization does not converge smoothly, try adjusting the optimization settings several times. For example, reduce `maxstep`, switch the optimization algorithm with `opt`, change the initial guess length with `spread`, or enable `no-update` so the optimization uses the true Hessian rather than an updated approximate Hessian.

---

### 3. Generate Instanton Sampled Configurations

After obtaining the optimized instanton structure, sample configurations around it.

**Related command:** `pyrinst-sampling`

**Example:**

```bash
pyrinst-sampling inst.pkl -T 300 -N 2048
```

This step is the same as in `Harm-FEP`. It generates the `simulation.pos` bead-configuration files and updates the harmonic energy stored in `inst.pkl`.

---

### 4. Evaluate Energies

Use the same procedure described above for `Harm-FEP`.

---

### 5. Compute the FEP Free-Energy Correction

Finally, use the computed bead energies to perform the last FEP evaluation and print the result.

**Related command:** `pyrinst-fep-eval`

**Example:**

```bash
pyrinst-fep-eval inst.pkl --prefix simulation.pos
```
