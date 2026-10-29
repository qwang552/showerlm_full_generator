#!/bin/bash
# Job script for generate_gpu.sub: run_gpu.sh <repo> <n> <out> <job> <seed0> <gpu: a40 | 1080>
set -euo pipefail
repo=$1 n=$2 out=$3 job=$4 seed0=$5 gpu=$6
mkdir -p "$(dirname "$out")"
echo "$(hostname)  $(nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1)  $(date)"
t0=$(date +%s)
python "$repo/generate.py" --n "$n" --out "$out" --seed $((seed0 + job)) --device "$gpu"
echo "finished $n showers in $(( $(date +%s) - t0 )) s"
