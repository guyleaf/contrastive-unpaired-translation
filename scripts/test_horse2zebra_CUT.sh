#!/usr/bin/env bash
set -e

for epoch in {5..400..5}
do
    python -m experiments horse2zebra test 0 \
                --which_epoch "$epoch" \
                --gpu_id 5
done
