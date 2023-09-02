#!/usr/bin/env bash



# conda activate sb-fbsde


python main.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --gpu $1

#python main.py --problem-name spiral --forward-net toy  --backward-net toy --dir output --log-tb


#python main.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb

#python main.py --problem-name gmm --forward-net linear  --backward-net toy --dir output --log-tb

#python main.py --problem-name checkerboard --forward-net toy  --backward-net toy --dir output --log-tb --gpu $1

