#!/usr/bin/env bash
set -eu

option=$1
gpu="${2:-0}"

for epoch in {5..400..5}
do
    python -m experiments fwid_image2weather test "$option" \
                --which_epoch "$epoch" \
                --gpu_id "$gpu"
done
