#!/usr/bin/env bash
set -e

root=$1
device="${2:-0}"
expName="${3:-weather_anti_uav_CUT_load_845_crop_768}"

if [ -z "$root" ]
then
    echo "Please enter the root folder of dataset or -h, --help for help."
    exit 1
fi

if [ "$root" = "-h" ] || [ "$root" = "--help" ]
then
    echo "Usage: $0 ROOT_FOLDER [DEVICE, e.g. 4] [EXP_NAME, e.g. weather_anti_uav_CUT_load_845_crop_768]"
    exit 0
fi

root="${root%/}"

echo "Root folder: $root"
echo "Device: $device"

outputPrefix="$root/stylized_cut_${expName}"
rm -rf "$outputPrefix" && mkdir -p "$outputPrefix"

mapfile -t datasets < <(find "$root/harmonized" -maxdepth 1 -type d -regex ".*/.*_fake_style$")
for dataset in "${datasets[@]}"
do
    name=$(basename "$dataset")
    output="$outputPrefix/$name"

    bash "$(dirname "$0")/translate_weather_anti_uav.sh" "$dataset" "$expName" "$output" "$device"

    input="${dataset%_fake_style}"
    output="$outputPrefix"
    cp -r "$input" "$output"
done
