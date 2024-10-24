#!/usr/bin/env bash
set -e

for epoch in {5..200..5}
do
    python -m experiments horse2zebra test 1 \
                --which_epoch "$epoch" \
                --gpu_id 6
done
