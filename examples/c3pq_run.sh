#!/usr/bin/env bash
# Run every compiled batch and collect its outputs under results/c3pq/<run>/.
#
#   examples/c3pq_build.sh  ->  staging/<batch>/<batch>.x
#   examples/c3pq_run.sh    ->  results/c3pq/<run>/<batch>/            (this script)
#   examples/c3pq_analyze.py -> TVD vs bound
#
# Each binary takes  <n_qubits> <output_dir>  and writes, per group:
#   probs   <group>.f64          the 2^n float64 probability vector
#   topk    <batch>.topk.tsv     the largest few probabilities, with the retained mass
#   parity  <batch>.parity.tsv   the CB survival, one scalar
#   always  <batch>.norm.tsv     the accumulated norm -- the cheap did-it-run check
# and prints one `entry` line per circuit with its wall time, kept as run.log.
#
# ONE RANK, DELIBERATELY. The batches are generated with --distributed_memory equal to
# the qubit count, so there is a single partition, no MPI_Alltoall, and the state vector
# never leaves the process. The harness asserts MPI_Comm_size == 1 rather than trusting
# the launch line, because running it under more ranks would not fail -- it would
# silently give every rank a private copy of the state and write whichever finished last.
# Width is scaled by using a bigger node, not more ranks; see --launcher slurm.
#
# Usage:
#   examples/c3pq_run.sh [--staging DIR] [--out DIR] [--batch NAME]... [-j N]
#                        [--threads N] [--launcher local|slurm] [--force]
#
# Environment:
#   MPIRUN      launcher for the local path  (default mpirun)
#   SBATCH_ARGS extra sbatch flags for the slurm path

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$REPO/staging"
OUT=""
MPIRUN="${MPIRUN:-mpirun}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

JOBS=1; THREADS=8; LAUNCHER=local; FORCE=0; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    --staging)  STAGING="$2"; shift 2 ;;
    --out)      OUT="$2";     shift 2 ;;
    --batch)    ONLY+=("$2"); shift 2 ;;
    -j)         JOBS="$2";    shift 2 ;;
    --threads)  THREADS="$2"; shift 2 ;;
    --launcher) LAUNCHER="$2"; shift 2 ;;
    --force)    FORCE=1;      shift ;;
    -h|--help)  sed -n '2,29p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

[ -f "$STAGING/jobs.json" ] || { echo "no $STAGING/jobs.json" >&2; exit 1; }
[ -n "$OUT" ] || OUT="$REPO/results/c3pq/$(basename "$STAGING")"

BATCHES=()
while read -r line; do BATCHES+=("$line"); done < <("$PYTHON" - "$STAGING/jobs.json" \
    "${ONLY[@]+"${ONLY[@]}"}" <<'PY'
import json, sys
jobs = json.load(open(sys.argv[1]))
only = set(sys.argv[2:])
for b in jobs["batches"]:
    if not only or b["name"] in only:
        print(b["name"], b["n_qubits"])
PY
)
[ ${#BATCHES[@]} -gt 0 ] || { echo "no matching batch" >&2; exit 1; }

echo "staging: $STAGING"
echo "output:  $OUT"
echo "run:     $LAUNCHER, ${#BATCHES[@]} batches, -j $JOBS, OMP_NUM_THREADS=$THREADS"
echo

export STAGING OUT MPIRUN THREADS FORCE

run_one() {
  local batch="$1" n="$2" t0 t1
  local bin="$STAGING/$batch/$batch.x" dir="$OUT/$batch"
  [ -x "$bin" ] || { echo "FAIL	$batch	no binary (run c3pq_build.sh)"; return 1; }
  if [ "$FORCE" = 0 ] && [ -f "$dir/$batch.norm.tsv" ]; then
    echo "skip	$batch	0"; return 0
  fi
  mkdir -p "$dir"
  t0=$(date +%s.%N)
  if ! OMP_NUM_THREADS="$THREADS" $MPIRUN -n 1 "$bin" "$n" "$dir" >"$dir/run.log" 2>&1
  then
    echo "FAIL	$batch	0"; tail -10 "$dir/run.log" >&2; return 1
  fi
  t1=$(date +%s.%N)
  echo "run	$batch	$(echo "$t1 - $t0" | bc)"
}
export -f run_one

if [ "$LAUNCHER" = local ]; then
  mkdir -p "$OUT"
  printf '%s\n' "${BATCHES[@]}" \
    | xargs -P "$JOBS" -n 2 bash -c 'run_one "$@"' _ \
    | tee "$OUT/run_times.tsv" \
    | awk -F'\t' '$1=="run"{n++; s+=$3; printf "  %-24s %7.2f s\n", $2, $3}
                  $1=="skip"{k++}
                  END {if(n) printf "  %d batches, %.1f s total\n", n, s;
                       if(k) printf "  %d already run (--force to rerun)\n", k}'
elif [ "$LAUNCHER" = slurm ]; then
  # One array task per batch. The array index selects a line of batches.txt, so the
  # same jobs.json drives the local and the cluster path with no second source of truth.
  mkdir -p "$OUT"
  printf '%s\n' "${BATCHES[@]}" > "$OUT/batches.txt"
  sbatch --array=1-${#BATCHES[@]} ${SBATCH_ARGS:-} \
         --export=ALL,STAGING="$STAGING",OUT="$OUT",THREADS="$THREADS" \
         "$REPO/examples/template_c3pq.sh"
else
  echo "--launcher must be local or slurm, got '$LAUNCHER'" >&2; exit 2
fi

echo
echo "next: $PYTHON examples/c3pq_analyze.py --staging $STAGING --results $OUT"
