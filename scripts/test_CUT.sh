#!/usr/bin/env bash
set -eu

exp=$1
option=$2
gpu="${3:-0}"

for epoch in {5..400..5}
do
    python -m experiments "$exp" test "$option" \
                --which_epoch "$epoch" \
                --gpu_id "$gpu"
done
