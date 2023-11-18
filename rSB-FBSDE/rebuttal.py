#!/usr/bin/env python3
import sys, os
import numpy as np



gpu = 0
if len(sys.argv) == 2:
    gpu = int(sys.argv[1])

print(f'set up GPU {gpu}')


for _ in range(5):
    seed = np.random.choice(np.arange(1000))   
    lr = np.random.choice([5e-4, 4e-4, 4e-4, 3e-4, 3e-4, 2e-4])
    reset_stage = np.random.choice([2, 3, 4])
    for interval in [10, 20, 40, 80]:
        FolderName = f'flower_spiral_lr_{lr}_reset_stage_{reset_stage}_bs_2000_seed_{seed}_time_{interval}'
        print(FolderName)
        try:
            os.system(f'python main.py --problem-name spiral --lr {lr} --interval {interval} --train-bs-x 2000 --gpu {gpu} --seed {seed} --reset-stage {reset_stage} --dir {FolderName} --log-tb')
        except:
            continue
