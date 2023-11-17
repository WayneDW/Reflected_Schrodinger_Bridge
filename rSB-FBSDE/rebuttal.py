#!/usr/bin/env python3
import sys, os
import numpy as np



gpu = 0
if len(sys.argv) == 2:
    gpu = int(sys.argv[1])

print(f'set up GPU {gpu}')

for _ in range(20):
    seed = np.random.choice(np.arange(1000))   
    lr = np.random.choice([4e-4, 3e-4])
    for interval in [10, 20, 40, 80]:
        FolderName = f'flower_spiral_lr_{lr}_bs_2000_seed_{seed}_time_{interval}'
        print(FolderName)
        try:
            os.system(f'python main.py --problem-name spiral --lr {lr} --interval {interval} --train-bs-x 2000 --gpu {gpu} --seed {seed} --dir {FolderName} --log-tb')
        except:
            continue
