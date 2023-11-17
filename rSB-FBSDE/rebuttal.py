#!/usr/bin/env python3
import sys, os
import numpy as np



gpu = 0
if len(sys.argv) == 2:
    gpu = int(sys.argv[1])

print(f'set up GPU {gpu}')

for _ in range(1):
    seed = np.random.choice(np.arange(1000))   

    for interval in [20, 80]:
        lr = 4e-4
        FolderName = f'flower_spiral_lr_{lr}_bs_2000_seed_{seed}_time_{interval}'
        print(FolderName)
        try:
            os.system(f'python main.py --problem-name spiral --lr {lr} --interval {interval} --train-bs-x 2000 --gpu {gpu} --seed {seed} --dir {FolderName} --log-tb')
        except:
            continue
