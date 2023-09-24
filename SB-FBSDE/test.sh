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


# bad back good forward
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_18268.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_47286.npz"
# bad back very good forward
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_59561.npz"
# bad back bad forward
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_27336.npz"
# bad back good forward

# a bit bad back good forward
#python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_12_seed_16407.npz"

# bad back slightly good forward
#python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_12_seed_21741.npz"

## very good example like it 90%
#python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_12_seed_27083.npz"
## good forward a bit OK back
#python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_12_seed_37586.npz"
## 85%
#python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_12_seed_42409.npz"
## 83%
#python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_12_seed_51585.npz"

## 78%
#python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_12_seed_67819.npz"
## 75%
#python Demo_generate_vector_field.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/moon-to-spiral_stage_12_seed_96326.npz"





# bad back good forward
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_18268.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_27336.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_47286.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_59561.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_78135.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_30181.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_36075.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_63467.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_8568.npz"

## OKish
python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_31203.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_27584.npz"
## OKish
python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_40947.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_17206.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_47968.npz"

## Perfect
## python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_51125.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_65766.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_6890.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_71829.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_78488.npz"

## nice back bad forward
##python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_86021.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_87756.npz"

## almost perfect
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_88243.npz"
#python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_92677.npz"
python Demo_generate_vector_field.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --load "checkpoint/0/debug/checkerboard_stage_12_seed_99465.npz"
