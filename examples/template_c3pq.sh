#!/bin/bash
# Slurm array task for one C-3PQ batch. Submitted by `c3pq_run.sh --launcher slurm`,
# which sets STAGING / OUT / THREADS in the environment and sizes the array to the number
# of batches. Modelled on the C-3PQ repo's experiments/template_dist.sh, with the two
# differences this pipeline requires:
#
#   * ONE task per batch, not one per partition. Batches are generated with
#     --distributed_memory equal to the qubit count, so a batch is a single partition and
#     the harness asserts MPI_Comm_size == 1. Extra ranks would each get a private copy of
#     the state vector and race to overwrite the same output files.
#   * CPU, not GPU. The study runs on the CPU path; -C gpu / --gpus-per-node are omitted
#     and the array dimension does the parallelism instead of the rank dimension.
#
# Width is scaled by asking for a fatter node (a 2^n float64 state vector is 8*2^n bytes,
# so n=30 needs 8 GB plus the working buffer), never by adding ranks.
#
# Edit the account, partition and time limit for your site, or pass them through
# SBATCH_ARGS:
#     SBATCH_ARGS="-A m1234 -q regular -t 04:00:00 -C cpu" \
#         examples/c3pq_run.sh --launcher slurm --threads 32

#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH -J c3pq_bounding
#SBATCH --output=%x_%A_%a.out
#SBATCH --error=%x_%A_%a.err
#SBATCH -t 02:00:00

set -euo pipefail

: "${STAGING:?c3pq_run.sh must export STAGING}"
: "${OUT:?c3pq_run.sh must export OUT}"
THREADS="${THREADS:-${SLURM_CPUS_PER_TASK:-8}}"

export OMP_NUM_THREADS="$THREADS"
export OMP_PROC_BIND=spread
export OMP_PLACES=cores

read -r BATCH NQUBITS < <(sed -n "${SLURM_ARRAY_TASK_ID}p" "$OUT/batches.txt")
BIN="$STAGING/$BATCH/$BATCH.x"
DIR="$OUT/$BATCH"
mkdir -p "$DIR"

echo "task $SLURM_ARRAY_TASK_ID: $BATCH (n=$NQUBITS, $THREADS threads) -> $DIR"
t0=$SECONDS
srun -n 1 -c "$THREADS" --cpu_bind=cores "$BIN" "$NQUBITS" "$DIR" | tee "$DIR/run.log"
echo "$BATCH finished in $((SECONDS - t0)) s"
