#!/usr/bin/env bash
# Generate, rename, and compile every staged batch into one binary per batch.
#
#   examples/c3pq_stage.py stage   ->  staging/<batch>/qasm/*.qasm
#   examples/c3pq_build.sh         ->  staging/<batch>/<batch>.x        (this script)
#   examples/c3pq_run.sh           ->  results/c3pq/<run>/
#
# Three steps per batch, in this order and for these reasons:
#
#   1. codegen   C-3PQ turns each QASM into C++. This is the expensive step and its cost
#                is per circuit and nearly independent of qubit count, which is the whole
#                reason batches exist -- and the reason every codegen is timed here. The
#                measured seconds/circuit printed at the end is the number that decides
#                whether the middle or the full config is affordable.
#   2. harness   c3pq_stage.py gives each circuit's symbols a private prefix (C-3PQ names
#                them after level/cluster indices, so every circuit exports the same ones
#                and no two could otherwise be linked) and writes the batch main.cpp.
#   3. compile   one mpicxx invocation over main.cpp plus every generated .cpp.
#
# Nothing under $C3PQ is modified: codegen writes into staging/<batch>/gen only.
#
# Usage:
#   examples/c3pq_build.sh [--staging DIR] [--batch NAME]... [-j N] [--force] [--dry-run]
#
# Environment:
#   C3PQ        C-3PQ checkout                 (default ~/Documents/Code/quantum)
#   MPICXX      compiler                       (default mpicxx)
#   OMP_FLAGS   OpenMP flags for this toolchain
#               (default: the Homebrew libomp incantation Apple clang needs)
#   CHUNK_SIZE  C-3PQ's vectorisation chunk    (default 64)

set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
STAGING="$REPO/staging"
C3PQ="${C3PQ:-$HOME/Documents/Code/quantum}"
MPICXX="${MPICXX:-mpicxx}"
CHUNK_SIZE="${CHUNK_SIZE:-64}"
OMP_FLAGS="${OMP_FLAGS:--I/opt/homebrew/opt/libomp/include/ -L/opt/homebrew/opt/libomp/lib/ -lomp -Xpreprocessor -fopenmp}"
PYTHON="${PYTHON:-$REPO/.venv/bin/python}"
[ -x "$PYTHON" ] || PYTHON=python3

JOBS=1; FORCE=0; DRY=0; ONLY=()
while [ $# -gt 0 ]; do
  case "$1" in
    --staging) STAGING="$2"; shift 2 ;;
    --batch)   ONLY+=("$2");  shift 2 ;;
    -j)        JOBS="$2";     shift 2 ;;
    --force)   FORCE=1;       shift ;;
    --dry-run) DRY=1;         shift ;;
    -h|--help) sed -n '2,32p' "${BASH_SOURCE[0]}"; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
done

GEN="$C3PQ/generator/test_circuits.py"
[ -f "$GEN" ] || { echo "C-3PQ generator not found at $GEN (set C3PQ=)" >&2; exit 1; }
[ -f "$STAGING/jobs.json" ] || { echo "no $STAGING/jobs.json -- run c3pq_stage.py stage" >&2; exit 1; }

# jobs.json is the single source of truth for what to build; read it with the same
# interpreter that wrote it rather than reimplementing the schema in shell.
read_jobs() { "$PYTHON" - "$STAGING/jobs.json" "$@" <<'PY'
import json, sys
jobs = json.load(open(sys.argv[1]))
what, only = sys.argv[2], set(sys.argv[3:])
batches = [b for b in jobs["batches"] if not only or b["name"] in only]
if what == "batches":
    for b in batches:
        print(b["name"], b["n_qubits"], len(b["entries"]))
elif what == "entries":
    cg = jobs["codegen"]
    for b in batches:
        for e in b["entries"]:
            print(b["name"], e["stem"], b["n_qubits"], cg["architecture"],
                  cg["threads"], cg["ranks"])
PY
}

echo "C-3PQ:    $C3PQ"
echo "staging:  $STAGING"
echo "compiler: $MPICXX -O2 -std=c++17 -DCHUNK_SIZE=$CHUNK_SIZE"
BATCHES=()
while read -r line; do BATCHES+=("$line"); done \
  < <(read_jobs batches "${ONLY[@]+"${ONLY[@]}"}")
[ ${#BATCHES[@]} -gt 0 ] || { echo "no matching batch" >&2; exit 1; }
printf '%s\n' "${BATCHES[@]}" | awk '{n+=$3} END {printf "batches:  %d (%d circuits)\n\n", NR, n}'

# ---------------------------------------------------------------------------
# 1. codegen -- the expensive step, hence the timing
# ---------------------------------------------------------------------------
# Exported so the xargs workers can see them.
export STAGING GEN PYTHON FORCE DRY

codegen_one() {
  local batch="$1" stem="$2" n="$3" arch="$4" threads="$5" ranks="$6"
  local gen="$STAGING/$batch/gen"
  local marker="$gen/${stem}_m.hpp"
  if [ "$FORCE" = 0 ] && [ -f "$marker" ]; then
    echo "skip	$batch	$stem	0"
    return 0
  fi
  mkdir -p "$gen"
  if [ "$DRY" = 1 ]; then echo "dry	$batch	$stem	0"; return 0; fi
  local t0 t1
  t0=$(date +%s.%N)
  # --distributed_memory == --qubits gives exactly one partition and one rank, so there
  # is no Alltoall and the state never leaves the process. That is what makes the
  # harness's per-circuit reset to |0...0> and its MPI_Comm_size == 1 assertion valid.
  if ! "$PYTHON" "$GEN" --input "$STAGING/$batch/qasm/$stem.qasm" \
        --output "$gen" --qubits "$n" --architecture "$arch" \
        --threads "$threads,1" --distributed_memory "$n" >"$gen/$stem.codegen.log" 2>&1
  then
    echo "FAIL	$batch	$stem	0" ; sed -n '$p' "$gen/$stem.codegen.log" >&2 ; return 1
  fi
  t1=$(date +%s.%N)
  echo "gen	$batch	$stem	$(echo "$t1 - $t0" | bc)"
}
export -f codegen_one

TIMES="$STAGING/codegen_times.tsv"
echo "==> codegen (-j $JOBS)"
read_jobs entries "${ONLY[@]+"${ONLY[@]}"}" \
  | xargs -P "$JOBS" -n 6 bash -c 'codegen_one "$@"' _ \
  | tee "$TIMES" \
  | awk -F'\t' '$1=="gen"{n++; s+=$4; if(n%25==0) printf "    %4d done, %.2f s/circuit\n", n, s/n}
                $1=="skip"{k++}
                END {if(n) printf "    %4d generated, %.3f s/circuit (%.1f s total)\n", n, s/n, s;
                     if(k) printf "    %4d already generated (--force to redo)\n", k}'

[ "$DRY" = 1 ] && { echo "(dry run: stopping before harness and compile)"; exit 0; }

# ---------------------------------------------------------------------------
# 2. harness -- symbol rename + main.cpp, in Python where the naming rules live
# ---------------------------------------------------------------------------
echo
echo "==> harness"
if [ ${#ONLY[@]} -gt 0 ]; then
  "$PYTHON" "$REPO/examples/c3pq_stage.py" harness --staging "$STAGING" \
      --batch "${ONLY[@]}"
else
  "$PYTHON" "$REPO/examples/c3pq_stage.py" harness --staging "$STAGING"
fi

# ---------------------------------------------------------------------------
# 3. compile -- one binary per batch
# ---------------------------------------------------------------------------
echo
echo "==> compile (-j $JOBS)"
export MPICXX CHUNK_SIZE OMP_FLAGS REPO

compile_one() {
  local batch="$1" root="$STAGING/$1" t0 t1
  local out="$root/$batch.x"
  if [ "$FORCE" = 0 ] && [ -x "$out" ] && [ "$out" -nt "$root/main.cpp" ]; then
    echo "skip	$batch	0"; return 0
  fi
  local srcs=()
  while read -r f; do srcs+=("$root/gen/$f"); done < "$root/sources.txt"
  t0=$(date +%s.%N)
  # -I gen: main.cpp includes each circuit's generated header by bare filename, and the
  # generated sources include each other the same way.
  if ! $MPICXX -O2 -std=c++17 -DCHUNK_SIZE="$CHUNK_SIZE" $OMP_FLAGS -I "$root/gen" \
        -o "$out" "$root/main.cpp" "${srcs[@]}" >"$root/compile.log" 2>&1; then
    echo "FAIL	$batch	0"; tail -20 "$root/compile.log" >&2; return 1
  fi
  t1=$(date +%s.%N)
  echo "cc	$batch	$(echo "$t1 - $t0" | bc)"
}
export -f compile_one

printf '%s\n' "${BATCHES[@]}" | awk '{print $1}' \
  | xargs -P "$JOBS" -n 1 bash -c 'compile_one "$@"' _ \
  | tee "$STAGING/compile_times.tsv" \
  | awk -F'\t' '$1=="cc"{n++; s+=$3; printf "    %-24s %6.1f s\n", $2, $3}
                $1=="skip"{k++}
                END {if(n) printf "    %d batches compiled, %.1f s total\n", n, s;
                     if(k) printf "    %d already up to date (--force to rebuild)\n", k}'

echo
echo "binaries:"
printf '%s\n' "${BATCHES[@]}" | awk -v s="$STAGING" '{printf "  %s/%s/%s.x\n", s, $1, $1}'
echo "next: examples/c3pq_run.sh --staging $STAGING"
