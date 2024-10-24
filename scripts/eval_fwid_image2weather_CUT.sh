#!/usr/bin/env bash
set -eu

imgs=$1
exp=$2
gpu=${3:-cuda:0}

# eval "$(conda shell.bash hook)"
# conda activate gen_eval

# for epoch in {5..400..5}
# do
#     python evaluation/eval_fid.py "$imgs" "$exp" \
#                 --batch-size 1 \
#                 --device "$gpu"
# done

python evaluation/eval_fid.py "$imgs" "$exp" \
                --batch-size 1 \
                --device "$gpu"
