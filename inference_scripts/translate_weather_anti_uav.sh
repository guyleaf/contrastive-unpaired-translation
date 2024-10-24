#!/usr/bin/env bash
set -eu

dataset=$1
out=$2
gpu="${3:-0}"
expSuffix="${4:-_load_845_crop_768_fdd}"

weathers=("foggy" "rainy" "snowy")

dataset="${dataset%/}"
out="${out%/}"

mkdir -p "$out"

for weather in "${weathers[@]}"
do
    root="$dataset/$weather"
    if [ ! -d "$root" ]; then
        echo "The '$root' folder doesn't exist. Skipped."
        continue
    fi

    # must be fixed corresponding to the checkpoint
    expName="weather_anti_uav_CUT_${weather}$expSuffix"

    # follow the instruction in https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/tree/master?tab=readme-ov-file#apply-a-pre-trained-model-cyclegan
    python test.py --dataroot "$root" \
        --name "$expName" \
        --model test \
        --dataset_mode single \
        --preprocess "scale_shortside" \
        --load_size 845 \
        --gpu_ids "$gpu"

    result="results/$expName"
    target="$out/$weather"
    rm -rf "$target"
    mv "$result/test_latest/images/fake" "$target"
    rm -r "$result"
done
