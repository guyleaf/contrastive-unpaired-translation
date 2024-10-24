#!/usr/bin/env bash
set -e

root=$1
metric="${2:-fdd}"
device="${3:-0}"

if [ -z "$root" ]
then
    echo "Please enter the root folder of dataset or -h, --help for help."
    exit 1
fi

if [ "$root" = "-h" ] || [ "$root" = "--help" ]
then
    echo "Usage: $0 ROOT_FOLDER [METRIC, e.g. fdd] [DEVICE, e.g. 4]"
    exit 0
fi

root="${root%/}"

echo "Root folder: $root"
echo "Device: $device"

datasets=("FWID" "Image2Weather")

expSuffix="_load_845_crop_768_$metric"
outputPrefix="$root/stylized_cut${expSuffix}_scaled"

rm -rf "$outputPrefix" && mkdir -p "$outputPrefix"

for dataset in "${datasets[@]}"
do
    input="$root/harmonized/${dataset}_fake_style"
    output="$outputPrefix/${dataset}_fake_style"

    bash "$(dirname "$0")/translate_weather_anti_uav.sh" "$input" "$output" "$device" "$expSuffix"

    input="$root/harmonized/${dataset}"
    output="$outputPrefix"
    cp -r "$input" "$output"
done
