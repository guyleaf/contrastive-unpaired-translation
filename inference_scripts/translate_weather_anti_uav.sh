#!/usr/bin/env bash
set -eu

dataset=$1
expName=$2
out=$3
gpu="${4:-0}"

dataset="${dataset%/}"
out="${out%/}"

mkdir -p "$out"

mapfile -t roots < <(find "$dataset" -maxdepth 1 ! -path "$dataset" -type d)
for root in "${roots[@]}"
do
    weather=$(basename "$root")

    name="${expName}_${weather}"
    if [ ! -d "checkpoints/$name" ]; then
        echo "The '$name' checkpoint doesn't exist. Skipped."
        continue
    fi

    python test.py --dataroot "$root" \
        --name "$name" \
        --model test \
        --num_test 10000000 \
        --dataset_mode single \
        --preprocess "none" \
        --gpu_ids "$gpu"

    result="results/$name"
    target="$out/$weather"
    rm -rf "$target"
    mv "$result/test_latest/images/fake" "$target"
    rm -r "$result"
done
