#!/usr/bin/env bash
set -eu

images=$1
exp=$2
epoch=$3
out=$4
gpu="${5:-0}"

images="${images%/}"
out="${out%/}"

python test.py --dataroot "$images" \
    --name "$exp" \
    --model test \
    --num_test 10000000 \
    --dataset_mode single \
    --preprocess "none" \
    --epoch "$epoch" \
    --gpu_ids "$gpu"

result="results/$exp/test_$epoch"
rm -rf "$out"
mkdir -p "$out"
mv "$result/images/real" "$result/images/fake" "$out"
# rm -r "$result"
