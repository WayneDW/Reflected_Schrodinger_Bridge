#!/usr/bin/env bash



# conda activate sb-fbsde


python main.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --ckpt-freq 2

#python main.py --problem-name smile-to-checkerboard --forward-net toy  --backward-net toy --dir output --log-tb


python main.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --ckpt-freq 2 

python main.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --ckpt-freq 2



# python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_18.npz"

# python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_16.npz"

# python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_14.npz"
