#!/usr/bin/env bash

# Perlmutter interactive XACC/TNQVM/ExaTN environment.
#
# Usage:
#
#   salloc -N 1 -C gpu -q interactive -t 01:00:00 -A nstaff
#   cd /pscratch/sd/n/nparr1/quantumsim
#   source scripts/activate_tnqvm.sh
#
# This script must be sourced, not executed, because it modifies the
# current shell environment.

#set -o pipefail

# ----------------------------------------------------------------------
# Repository and installation locations
# ----------------------------------------------------------------------

export QUANTUMSIM_ROOT="/pscratch/sd/n/nparr1/quantumsim"
export STACK_REPO="/pscratch/sd/n/nparr1/tnqvm-exatn-hpc"

export STACK_CONFIG="$STACK_REPO/config/perlmutter-gpu.env"
export STACK_COMMON_SH="$STACK_REPO/scripts/lib/common.sh"

# ----------------------------------------------------------------------
# Error helper
# ----------------------------------------------------------------------

tnqvm_fail() {
    echo "[activate_tnqvm] ERROR: $*" >&2
    return 1
}

# ----------------------------------------------------------------------
# Validate required paths
# ----------------------------------------------------------------------

[ -d "$QUANTUMSIM_ROOT" ] || {
    tnqvm_fail "quantumsim repository not found: $QUANTUMSIM_ROOT"
    return 1
}

[ -d "$STACK_REPO" ] || {
    tnqvm_fail "stack repository not found: $STACK_REPO"
    return 1
}

[ -f "$STACK_CONFIG" ] || {
    tnqvm_fail "stack configuration not found: $STACK_CONFIG"
    return 1
}

[ -f "$STACK_COMMON_SH" ] || {
    tnqvm_fail "common.sh not found: $STACK_COMMON_SH"
    return 1
}

# ----------------------------------------------------------------------
# Clear stale preload settings
# ----------------------------------------------------------------------

unset LD_PRELOAD 2>/dev/null || true

# ----------------------------------------------------------------------
# Cray, MPI, and runtime configuration
# ----------------------------------------------------------------------

export CRAY_CPU_TARGET="x86-64"
export CRAYPE_LINK_TYPE="dynamic"

export MPI_LIB="MPICH"
export MPI_ROOT_DIR="/opt/cray/pe/mpich/9.0.1/ofi/gnu/12.3"

export MKLROOT="${MKLROOT:-/opt/intel/oneapi/mkl/2023.2.0}"

# ----------------------------------------------------------------------
# Load stack modules and configuration
# ----------------------------------------------------------------------

source "$STACK_COMMON_SH"

echo "[activate_tnqvm] Loading stack configuration:"
echo "  $STACK_CONFIG"

if ! load_config "$STACK_CONFIG"; then
    tnqvm_fail "load_config failed"
    return 1
fi

if [ -z "${INSTALL_ROOT:-}" ]; then
    tnqvm_fail "INSTALL_ROOT was not set by the stack configuration"
    return 1
fi

export XACC_ROOT="$INSTALL_ROOT/xacc"
export EXATN_ROOT="$INSTALL_ROOT/exatn"
export TNQVM_ROOT="$INSTALL_ROOT/tnqvm"

[ -d "$XACC_ROOT" ] || {
    tnqvm_fail "XACC installation not found: $XACC_ROOT"
    return 1
}

[ -d "$EXATN_ROOT" ] || {
    tnqvm_fail "ExaTN installation not found: $EXATN_ROOT"
    return 1
}

[ -f "$XACC_ROOT/_pyxacc.so" ] || {
    tnqvm_fail "XACC Python extension not found: $XACC_ROOT/_pyxacc.so"
    return 1
}

# ----------------------------------------------------------------------
# Locate Intel MKL
# ----------------------------------------------------------------------

MKL_LIBRARY_DIR=""

for candidate in \
    "$MKLROOT/lib/intel64" \
    "$MKLROOT/lib"
do
    if [ -d "$candidate" ]; then
        MKL_LIBRARY_DIR="$candidate"
        break
    fi
done

if [ -z "$MKL_LIBRARY_DIR" ]; then
    tnqvm_fail "MKL library directory not found under MKLROOT=$MKLROOT"
    return 1
fi

export MKL_LIBRARY_DIR

MKL_RT_LIBRARY=""

for candidate in \
    "$MKL_LIBRARY_DIR/libmkl_rt.so.2" \
    "$MKL_LIBRARY_DIR/libmkl_rt.so"
do
    if [ -e "$candidate" ]; then
        MKL_RT_LIBRARY="$(readlink -f "$candidate")"
        break
    fi
done

if [ -z "$MKL_RT_LIBRARY" ]; then
    MKL_RT_LIBRARY="$(
        find "$MKLROOT" \
            \( -type f -o -type l \) \
            -name 'libmkl_rt.so*' \
            2>/dev/null |
        head -1
    )"
fi

if [ -z "$MKL_RT_LIBRARY" ] || [ ! -e "$MKL_RT_LIBRARY" ]; then
    tnqvm_fail "libmkl_rt.so was not found under MKLROOT=$MKLROOT"
    return 1
fi

export MKL_RT_LIBRARY

# Verify that the component library required by TNQVM exists.
MKL_LP64_LIBRARY=""

for candidate in \
    "$MKL_LIBRARY_DIR/libmkl_intel_lp64.so.2" \
    "$MKL_LIBRARY_DIR/libmkl_intel_lp64.so"
do
    if [ -e "$candidate" ]; then
        MKL_LP64_LIBRARY="$(readlink -f "$candidate")"
        break
    fi
done

if [ -z "$MKL_LP64_LIBRARY" ]; then
    tnqvm_fail "libmkl_intel_lp64.so was not found in $MKL_LIBRARY_DIR"
    return 1
fi

export MKL_LP64_LIBRARY

# ----------------------------------------------------------------------
# Locate optional Intel compiler runtime directories
# ----------------------------------------------------------------------

INTEL_RUNTIME_PATH=""

for candidate in \
    "/opt/intel/oneapi/compiler/2023.2.0/linux/compiler/lib/intel64_lin" \
    "/opt/intel/oneapi/compiler/2023.2.0/linux/lib" \
    "/opt/intel/oneapi/compiler/latest/linux/compiler/lib/intel64_lin" \
    "/opt/intel/oneapi/compiler/latest/linux/lib"
do
    if [ -d "$candidate" ]; then
        INTEL_RUNTIME_PATH="$candidate${INTEL_RUNTIME_PATH:+:$INTEL_RUNTIME_PATH}"
    fi
done

# ----------------------------------------------------------------------
# Configure executable, Python, and shared-library paths
# ----------------------------------------------------------------------

export PATH="$XACC_ROOT/bin:${PATH:-}"

export PYTHONPATH="$XACC_ROOT:${PYTHONPATH:-}"

for python_dir in \
    "$XACC_ROOT/python" \
    "$XACC_ROOT/python3" \
    "$XACC_ROOT/lib/python3.13/site-packages" \
    "$XACC_ROOT/lib64/python3.13/site-packages"
do
    if [ -d "$python_dir" ]; then
        export PYTHONPATH="$python_dir:$PYTHONPATH"
    fi
done

NEW_LIBRARY_PATH="$MKL_LIBRARY_DIR"

if [ -n "$INTEL_RUNTIME_PATH" ]; then
    NEW_LIBRARY_PATH="$NEW_LIBRARY_PATH:$INTEL_RUNTIME_PATH"
fi

NEW_LIBRARY_PATH="$NEW_LIBRARY_PATH:$XACC_ROOT/lib"
NEW_LIBRARY_PATH="$NEW_LIBRARY_PATH:$EXATN_ROOT/lib"

if [ -d "$TNQVM_ROOT/lib" ]; then
    NEW_LIBRARY_PATH="$NEW_LIBRARY_PATH:$TNQVM_ROOT/lib"
fi

export LD_LIBRARY_PATH="$NEW_LIBRARY_PATH:${LD_LIBRARY_PATH:-}"

# MKL runtime choice. GNU threading is appropriate for PrgEnv-gnu.
export MKL_INTERFACE_LAYER="LP64"
export MKL_THREADING_LAYER="GNU"

# ----------------------------------------------------------------------
# Activate quantumsim virtual environment only when Python versions match
# ----------------------------------------------------------------------

STACK_PYTHON="$(command -v python)"

STACK_PYTHON_VERSION="$(
    "$STACK_PYTHON" -c \
        'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")'
)"

VENV_ACTIVATE="$QUANTUMSIM_ROOT/.venv/bin/activate"

if [ -f "$VENV_ACTIVATE" ]; then
    VENV_PYTHON_VERSION="$(
        "$QUANTUMSIM_ROOT/.venv/bin/python" -c \
            'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")' \
            2>/dev/null || true
    )"

    if [ "$VENV_PYTHON_VERSION" = "$STACK_PYTHON_VERSION" ]; then
        source "$VENV_ACTIVATE"
        echo "[activate_tnqvm] Activated quantumsim .venv"
    else
        echo "[activate_tnqvm] Not activating incompatible .venv"
        echo "  Stack Python: $STACK_PYTHON_VERSION"
        echo "  .venv Python: ${VENV_PYTHON_VERSION:-unknown}"
        echo
        echo "Recreate it after this script succeeds:"
        echo "  rm -rf $QUANTUMSIM_ROOT/.venv"
        echo "  python -m venv --system-site-packages $QUANTUMSIM_ROOT/.venv"
        echo "  source $QUANTUMSIM_ROOT/.venv/bin/activate"
        echo "  python -m pip install -e $QUANTUMSIM_ROOT"
        echo
    fi
fi

# ----------------------------------------------------------------------
# Diagnostics
# ----------------------------------------------------------------------

echo
echo "[activate_tnqvm] Environment configuration"
echo "  QUANTUMSIM_ROOT  = $QUANTUMSIM_ROOT"
echo "  STACK_REPO       = $STACK_REPO"
echo "  INSTALL_ROOT     = $INSTALL_ROOT"
echo "  XACC_ROOT        = $XACC_ROOT"
echo "  EXATN_ROOT       = $EXATN_ROOT"
echo "  TNQVM_ROOT       = $TNQVM_ROOT"
echo "  MKLROOT          = $MKLROOT"
echo "  MKL_LIBRARY_DIR  = $MKL_LIBRARY_DIR"
echo "  MKL runtime      = $MKL_RT_LIBRARY"
echo "  MKL LP64         = $MKL_LP64_LIBRARY"
echo "  Python           = $(command -v python)"
python --version
echo

# ----------------------------------------------------------------------
# Verify the complete MKL runtime, not an individual component library
# ----------------------------------------------------------------------

echo "[activate_tnqvm] Verifying MKL runtime..."

if ! python - <<'PY'
import ctypes

library = "libmkl_rt.so.2"
ctypes.CDLL(library, mode=ctypes.RTLD_GLOBAL)

print("MKL runtime load successful:", library)
PY
then
    tnqvm_fail "Python could not load libmkl_rt.so.2"
    return 1
fi

# ----------------------------------------------------------------------
# Test XACC import and plugin initialization
# ----------------------------------------------------------------------

echo
echo "[activate_tnqvm] Testing XACC import and plugin initialization..."

if ! python - <<'PY'
import ctypes
import sys

# Make MKL symbols globally available before XACC initializes plugins.
ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)

print("Python executable:", sys.executable)
print("Python version:", sys.version.splitlines()[0])

import xacc

print("XACC import successful")
print("XACC module:", getattr(xacc, "__file__", "<unknown>"))
PY
then
    tnqvm_fail "Python could not import and initialize XACC"
    return 1
fi

# ----------------------------------------------------------------------
# Inspect installed TNQVM plugin libraries
# ----------------------------------------------------------------------

echo
echo "[activate_tnqvm] Installed TNQVM/ExaTN plugin libraries:"

find \
    "$XACC_ROOT" \
    "$TNQVM_ROOT" \
    -type f \
    \( -name '*tnqvm*.so*' -o -name '*exatn*.so*' \) \
    2>/dev/null |
sort |
sed 's/^/  /' |
head -50

# ----------------------------------------------------------------------
# Test accelerator names
# ----------------------------------------------------------------------

echo
echo "[activate_tnqvm] Checking accelerator availability..."

python - <<'PY'
import ctypes

ctypes.CDLL("libmkl_rt.so.2", mode=ctypes.RTLD_GLOBAL)

import xacc

accelerator_names = [
    "tnqvm",
    "tnqvm:exatn",
    "exatn",
    "tnqvm_exatn",
    "tnqvm_exatn_dm",
]

for name in accelerator_names:
    try:
        accelerator = xacc.getAccelerator(name)
    except Exception as exc:
        print(f"  {name}: unavailable ({exc})")
    else:
        try:
            resolved_name = accelerator.name()
        except Exception:
            resolved_name = "<name unavailable>"

        print(f"  {name}: available ({resolved_name})")
PY

echo
echo "[activate_tnqvm] Environment ready."

set +u