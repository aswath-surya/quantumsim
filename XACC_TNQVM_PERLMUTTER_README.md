# Running XACC, TNQVM, and ExaTN from Python on Perlmutter

This guide describes how to configure an interactive Perlmutter session so that Python can import XACC and use the TNQVM ExaTN backend.

The setup assumes the following locations:

```text
quantumsim repository:
/pscratch/sd/n/nparr1/quantumsim

TNQVM/ExaTN build repository:
/pscratch/sd/n/nparr1/tnqvm-exatn-hpc

Installed XACC/TNQVM/ExaTN stack:
/pscratch/sd/n/nparr1/tnqvm-stack-install
```

The environment setup is handled by:

```text
quantumsim/scripts/activate_tnqvm.sh
```

---

## 1. Start an interactive GPU allocation

Log in to Perlmutter:

```bash
ssh nparr1@perlmutter.nersc.gov
```

Request an interactive GPU node:

```bash
salloc -N 1 -C gpu -q interactive -t 00:30:00 -A nstaff
```

After the allocation is granted, the prompt should change to a compute node such as:

```text
nparr1@nid001013
```

Move into the `quantumsim` repository:

```bash
cd /pscratch/sd/n/nparr1/quantumsim
```

The activation script uses paths relative to this installation, so it should be sourced from the repository:

```bash
source scripts/activate_tnqvm.sh
```

Do not run:

```bash
./scripts/activate_tnqvm.sh
```

The script must be sourced so that its exported environment variables remain active in the current shell.

---

## 2. What the activation script configures

The activation script performs the following operations:

1. Loads the Perlmutter TNQVM stack configuration.
2. Loads the required compiler, CUDA, CMake, Cray, and Python modules.
3. Sets the Cray dynamic-linking configuration required for XACC plugins.
4. Configures MPI paths.
5. Configures Intel MKL.
6. Adds the XACC, TNQVM, and ExaTN libraries to the runtime search path.
7. Adds the XACC Python package to `PYTHONPATH`.
8. Loads the MKL runtime before XACC initializes its plugin registry.
9. Activates the local Python virtual environment when its Python version matches the XACC build.
10. Tests XACC, TNQVM, and the ExaTN visitor.

A successful activation should end with output similar to:

```text
XACC import successful
TNQVM accelerator: tnqvm
TNQVM ExaTN accelerator: tnqvm

[activate_tnqvm] Environment ready.
```

The valid ExaTN accelerator construction is:

```python
accelerator = xacc.getAccelerator(
    "tnqvm",
    {"tnqvm-visitor": "exatn"},
)
```

The shorthand below may also work:

```python
accelerator = xacc.getAccelerator("tnqvm:exatn")
```

Names such as the following are not standalone accelerator registry names:

```text
exatn
tnqvm_exatn
tnqvm_exatn_dm
```

They may correspond to plugin filenames, but they should not be passed directly to `xacc.getAccelerator()`.

---

## 3. Create the Python virtual environment

XACC was built using the Perlmutter Python module loaded by the TNQVM stack. The Python virtual environment must use the same Python major and minor version.

At the time of this setup, XACC was built against Python 3.13.

After successfully sourcing `activate_tnqvm.sh`, recreate the virtual environment:

```bash
cd /pscratch/sd/n/nparr1/quantumsim

rm -rf .venv

python --version
```

Confirm that the reported version is Python 3.13 before continuing.

Create the environment:

```bash
python -m venv --system-site-packages .venv
```

Activate it:

```bash
source .venv/bin/activate
```

Verify the interpreter:

```bash
which python
python --version
```

The executable should be:

```text
/pscratch/sd/n/nparr1/quantumsim/.venv/bin/python
```

The version should still be Python 3.13.

Upgrade the Python packaging tools:

```bash
python -m pip install --upgrade pip setuptools wheel
```

Install `quantumsim` or `proxysim` in editable mode:

```bash
python -m pip install -e .
```

Use:

```bash
python -m pip
```

instead of `pip` or `pip3`. This ensures packages are installed into the Python interpreter currently active in the environment.

---

## 4. Verify the complete Python stack

After installing the repository, run:

```bash
python - <<'PY'
import ctypes
import sys

ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)

import xacc
import proxysim

print("Python:", sys.executable)
print("Python version:", sys.version)
print("XACC:", xacc.__file__)
print("proxysim:", proxysim.__file__)

accelerator = xacc.getAccelerator(
    "tnqvm",
    {"tnqvm-visitor": "exatn"},
)

print("Accelerator:", accelerator.name())
PY
```

Expected output should include:

```text
XACC: /pscratch/sd/n/nparr1/tnqvm-stack-install/xacc/xacc.py
proxysim: /pscratch/sd/n/nparr1/quantumsim/proxysim/...
Accelerator: tnqvm
```

---

## 5. Normal interactive workflow

Once the Python 3.13 virtual environment has been created, the normal workflow is:

```bash
ssh nparr1@perlmutter.nersc.gov
```

Request a node:

```bash
salloc -N 1 -C gpu -q interactive -t 00:30:00 -A nstaff
```

Move into the repository:

```bash
cd /pscratch/sd/n/nparr1/quantumsim
```

Load the environment:

```bash
source scripts/activate_tnqvm.sh
```

The script should automatically activate `.venv` when its Python version matches the XACC build.

Run a Python program:

```bash
python examples/brickwork_1C_no_noise.py
```

Other examples may be run in the same way:

```bash
python examples/cycle_benchmarking.py
python examples/QAOA_n36.py
```

Adjust the filenames to match the files present in the repository.

---

## 6. Minimal XACC and ExaTN test

Create a file named `examples/test_xacc_exatn.py`:

```python
import ctypes

# Make MKL symbols globally available before XACC initializes plugins.
ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)

import xacc

accelerator = xacc.getAccelerator(
    "tnqvm",
    {
        "tnqvm-visitor": "exatn",
    },
)

print("Loaded accelerator:", accelerator.name())
```

Run it from an interactive allocation:

```bash
cd /pscratch/sd/n/nparr1/quantumsim
source scripts/activate_tnqvm.sh
python examples/test_xacc_exatn.py
```

Expected output:

```text
Loaded accelerator: tnqvm
```

---

## 7. Important environment variables

The activation script defines the following paths:

```bash
QUANTUMSIM_ROOT=/pscratch/sd/n/nparr1/quantumsim
STACK_REPO=/pscratch/sd/n/nparr1/tnqvm-exatn-hpc
INSTALL_ROOT=/pscratch/sd/n/nparr1/tnqvm-stack-install
```

The installed components are expected at:

```bash
XACC_ROOT=$INSTALL_ROOT/xacc
EXATN_ROOT=$INSTALL_ROOT/exatn
TNQVM_ROOT=$INSTALL_ROOT/tnqvm
```

The MKL installation is expected at:

```bash
MKLROOT=/opt/intel/oneapi/mkl/2023.2.0
```

The relevant MKL library directory is:

```bash
$MKLROOT/lib/intel64
```

The activation script adds the following directories to the runtime environment:

```text
XACC Python package:
$XACC_ROOT

XACC libraries:
$XACC_ROOT/lib

XACC plugins:
$XACC_ROOT/plugins

ExaTN libraries:
$EXATN_ROOT/lib

TNQVM libraries:
$TNQVM_ROOT/lib

TNQVM plugins:
$TNQVM_ROOT/plugins

Intel MKL:
$MKLROOT/lib/intel64
```

---

## 8. Why MKL is loaded explicitly

The TNQVM ExaTN density-matrix plugin depends on Intel MKL.

Without MKL in the runtime library path, XACC may report:

```text
Failure initializing XACC Plugin Registry
Could not load tnqvm_exatn_dm
libmkl_intel_lp64.so.2: cannot open shared object file
```

The activation script adds the MKL library directory to `LD_LIBRARY_PATH`.

It also validates the unified MKL runtime:

```text
libmkl_rt.so.2
```

The component library:

```text
libmkl_intel_lp64.so.2
```

should not be loaded by itself using `ctypes.CDLL`. It depends on additional MKL components and may produce an error such as:

```text
undefined symbol: mkl_blas_dgemm
```

The correct standalone MKL validation is:

```python
import ctypes

ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)
```

---

## 9. Python-version compatibility

The XACC Python extension was built with a specific Python interpreter.

Inspect the XACC build configuration with:

```bash
grep -E \
    'Python.*EXECUTABLE|PYTHON.*EXECUTABLE' \
    /pscratch/sd/n/nparr1/src/tnqvm-stack/xacc/build/CMakeCache.txt
```

Example:

```text
_Python_EXECUTABLE:INTERNAL=/global/common/software/nersc/pe/conda-envs/26.1.0/python-3.13/nersc-python/bin/python3.13
```

The virtual environment must use that same Python major and minor version.

A Python 3.11 environment should not be used with an XACC extension built for Python 3.13.

Check the virtual environment version with:

```bash
/pscratch/sd/n/nparr1/quantumsim/.venv/bin/python --version
```

If it does not match the stack Python, recreate it after sourcing the activation script:

```bash
rm -rf .venv
python -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install -e .
```

---

## 10. Troubleshooting

### `scripts/activate_tnqvm.sh: No such file or directory`

This usually means the current directory is not the `quantumsim` repository.

Run:

```bash
cd /pscratch/sd/n/nparr1/quantumsim
source scripts/activate_tnqvm.sh
```

Alternatively, source it by absolute path:

```bash
source /pscratch/sd/n/nparr1/quantumsim/scripts/activate_tnqvm.sh
```

---

### `Permission denied`

The script does not need executable permission when it is sourced.

Use:

```bash
source scripts/activate_tnqvm.sh
```

Do not use:

```bash
./scripts/activate_tnqvm.sh
```

If direct execution is needed for another reason:

```bash
chmod +x scripts/activate_tnqvm.sh
```

However, direct execution will not preserve exported environment variables in the parent shell.

---

### `xacc: NOT IMPORTABLE`

Check that the XACC Python path is present:

```bash
echo "$PYTHONPATH"
```

It should include:

```text
/pscratch/sd/n/nparr1/tnqvm-stack-install/xacc
```

Test directly:

```bash
python -c "import xacc; print(xacc.__file__)"
```

---

### `CXXABI_1.3.15 not found`

This previously occurred when a Python 3.11 Conda environment loaded an older `libstdc++.so.6`.

The correct solution is to use the Python version and compiler environment against which XACC was built.

Check the interpreter:

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
```

Check the XACC build interpreter:

```bash
grep '_Python_EXECUTABLE' \
    /pscratch/sd/n/nparr1/src/tnqvm-stack/xacc/build/CMakeCache.txt
```

Recreate `.venv` with the stack Python if the versions differ.

---

### `libmkl_intel_lp64.so.2: cannot open shared object file`

Confirm that MKL is available:

```bash
ls -l /opt/intel/oneapi/mkl/2023.2.0/lib/intel64/libmkl_intel_lp64.so.2
```

Confirm that the MKL directory is in `LD_LIBRARY_PATH`:

```bash
echo "$LD_LIBRARY_PATH" | tr ':' '\n' | grep mkl
```

Validate the MKL runtime:

```bash
python - <<'PY'
import ctypes
ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)
print("MKL runtime available")
PY
```

---

### `undefined symbol: mkl_blas_dgemm`

This occurs when trying to load `libmkl_intel_lp64.so.2` directly.

Do not use:

```python
ctypes.CDLL("libmkl_intel_lp64.so.2")
```

Use:

```python
ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)
```

---

### Invalid accelerator errors

The following are not valid accelerator names in this installation:

```text
exatn
tnqvm_exatn
tnqvm_exatn_dm
```

Use:

```python
xacc.getAccelerator(
    "tnqvm",
    {"tnqvm-visitor": "exatn"},
)
```

---

### `CRAYBLAS_WARNING: Application linked against multiple cray-libsci libraries`

The warning may appear after XACC and ExaTN load successfully:

```text
[CRAYBLAS_WARNING] Application linked against multiple cray-libsci libraries
```

This indicates that multiple BLAS implementations or multiple Cray LibSci variants are present in the process.

If the calculation runs correctly, this warning is not necessarily fatal. However, numerical behavior and performance should be validated.

Inspect loaded BLAS and MKL libraries with:

```bash
ldd /pscratch/sd/n/nparr1/tnqvm-stack-install/xacc/_pyxacc.so \
    | grep -E 'blas|mkl|sci'
```

Plugin dependencies can be inspected with:

```bash
ldd \
/pscratch/sd/n/nparr1/tnqvm-stack-install/tnqvm/plugins/libtnqvm-exatn.so \
    | grep -E 'blas|mkl|sci|not found'
```

---

### Cray CPE module warning

The following warning may appear while loading modules:

```text
Unloading the cpe module is insufficient to restore the system defaults.
Please run:
source /opt/cray/pe/cpe/25.09/restore_lmod_system_defaults.sh
```

The warning does not necessarily prevent the environment from loading.

The important success criteria are:

```text
XACC import successful
TNQVM accelerator: tnqvm
TNQVM ExaTN accelerator: tnqvm
Environment ready
```

If module loading actually fails, start a fresh allocation and avoid manually loading additional compiler or Python modules before sourcing the activation script.

---

## 11. Inspect installed plugins

List installed TNQVM and ExaTN plugins:

```bash
find \
    /pscratch/sd/n/nparr1/tnqvm-stack-install/xacc \
    /pscratch/sd/n/nparr1/tnqvm-stack-install/tnqvm \
    -type f \
    \( -name '*tnqvm*.so*' -o -name '*exatn*.so*' \) \
    | sort
```

Expected plugin files include:

```text
libtnqvm.so
libtnqvm-exatn.so
libtnqvm-exatn-dm.so
libtnqvm-exatn-gen.so
libtnqvm-exatn-mpo.so
libtnqvm-exatn-mps.so
```

These are plugin library filenames. They are not necessarily the names used in `xacc.getAccelerator()`.

---

## 12. Running batch jobs

The same environment setup can be used in an `sbatch` script.

Example:

```bash
#!/usr/bin/env bash

#SBATCH -N 1
#SBATCH -C gpu
#SBATCH -q regular
#SBATCH -t 00:30:00
#SBATCH -A nstaff
#SBATCH --gpus-per-node=4
#SBATCH -J proxysim-exatn
#SBATCH -o logs/%x-%j.out
#SBATCH -e logs/%x-%j.err

set -euo pipefail

cd /pscratch/sd/n/nparr1/quantumsim

source scripts/activate_tnqvm.sh

python examples/brickwork_1C_no_noise.py
```

Create the log directory before submitting:

```bash
mkdir -p logs
```

Submit the job:

```bash
sbatch run_exatn.sbatch
```

Monitor it:

```bash
squeue -u "$USER"
```

Cancel it if needed:

```bash
scancel JOB_ID
```

---

## 13. Recommended validation sequence

Before running a large simulation, validate the environment in this order.

### Step 1: Python

```bash
python -c "import sys; print(sys.executable); print(sys.version)"
```

### Step 2: MKL

```bash
python - <<'PY'
import ctypes
ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)
print("MKL OK")
PY
```

### Step 3: XACC

```bash
python -c "import xacc; print('XACC OK:', xacc.__file__)"
```

### Step 4: TNQVM

```bash
python - <<'PY'
import ctypes
ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)

import xacc

accelerator = xacc.getAccelerator("tnqvm")
print("TNQVM:", accelerator.name())
PY
```

### Step 5: TNQVM with ExaTN

```bash
python - <<'PY'
import ctypes
ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)

import xacc

accelerator = xacc.getAccelerator(
    "tnqvm",
    {"tnqvm-visitor": "exatn"},
)

print("TNQVM/ExaTN:", accelerator.name())
PY
```

### Step 6: proxysim

```bash
python -c "import proxysim; print('proxysim:', proxysim.__file__)"
```

### Step 7: Small test circuit

Run the smallest available circuit before attempting large tensor-network calculations:

```bash
python examples/brickwork_1C_no_noise.py
```

---

## 14. Complete daily workflow

After the environment has been installed correctly, the typical workflow is:

```bash
ssh nparr1@perlmutter.nersc.gov

salloc -N 1 -C gpu -q interactive -t 00:30:00 -A nstaff

cd /pscratch/sd/n/nparr1/quantumsim

source scripts/activate_tnqvm.sh

python examples/brickwork_1C_no_noise.py
```

The activation script should report:

```text
XACC import successful
TNQVM accelerator: tnqvm
TNQVM ExaTN accelerator: tnqvm
Environment ready
```

At that point, the current shell is configured to run Python programs using XACC, TNQVM, and the ExaTN tensor-network backend.
