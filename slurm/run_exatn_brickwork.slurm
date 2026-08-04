#!/bin/bash
# Slurm script for running ExaTN brickwork validation
# This script performs a smoke test on a single small circuit.

#SBATCH --job-name=exatn_smoke
#SBATCH --output=logs/exatn_smoke_%j.out
#SBATCH --error=logs/exatn_smoke_%j.err
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:30:00
#SBATCH --partition=debug

# Load environment (adjust based on actual Perlmutter/XACC setup)
# module load xacc/exatn
# source /path/to/xacc/setup.sh

mkdir -p logs
mkdir -p results_exatn_test

echo "--- Environment Check ---"
python run_exatn_bank.py --check-environment

echo "--- Running Smoke Test ---"
# Run one small circuit from the test bank
python run_exatn_bank.py \
  --qasm-bank qasm_bank_exatn_test/brickwork \
  --output-dir results_exatn_test \
  --shots 1000 \
  --pattern "*brickwork_N2_depth1_instance0.qasm"

echo "--- Validating ---"
# We assume a reference set exists for these small circuits
python verify_exatn_brickwork.py \
  --qasm-bank qasm_bank_exatn_test/brickwork \
  --exatn-results results_exatn_test \
  --reference-results results_reference_test \
  --shots 1000 \
  --report logs/smoke_validation.json

if [ $? -eq 0 ]; then
    echo "Validation Successful"
else
    echo "Validation Failed"
    exit 1
fi
