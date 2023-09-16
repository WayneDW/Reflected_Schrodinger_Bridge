#!/usr/bin/env bash



# conda activate sb-fbsde


python main.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --ckpt-freq 2

#python main.py --problem-name smile-to-checkerboard --forward-net toy  --backward-net toy --dir output --log-tb


python main.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --ckpt-freq 2 

python main.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --ckpt-freq 2

### perfect
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_51484.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_42637.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_76588.npz"

### OKish....
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_36670.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_61722.npz"
# python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_19310.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_86286.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_9582.npz"

### Bad
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_49518.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_55251.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_59691.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_67955.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_69213.npz"
# python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_72577.npz"
#python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_12_seed_85868.npz"


# python Demo_generate_vector_field.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/gmm_stage_16.npz"
# python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_14.npz"

#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_18.npz"
python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/"



