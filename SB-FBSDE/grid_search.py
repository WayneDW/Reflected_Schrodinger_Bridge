#!/usr/bin/env python3
import os, sys, re

import numpy as np
import shutil
import termcolor

def myColor(content, color): return termcolor.colored(str(content), color, attrs=["bold"])

print(myColor('Set export MKL_SERVICE_FORCE_INTEL=1 before Calling it', 'red'))

gpu = 0
if len(sys.argv) > 1:
    gpu = sys.argv[1]
    print(myColor(f"Applying the custom GPU {gpu}", 'green'))
else:
    print(myColor(f"Running using the default GPU 0", 'yellow'))

#print(myColor('Set export MKL_SERVICE_FORCE_INTEL=1 before Calling it', 'red'))
#os.system('export MKL_SERVICE_FORCE_INTEL=1')

for i in range(20):
    name = np.random.choice(['moon-to-spiral', 'checkerboard'])
    lr = np.random.choice([1e-3, 6e-4, 3e-4, 1e-4, 6e-4, 3e-4, 1e-4, 6e-5])
    lr_gamma = np.random.choice([0.9, 0.8])
    t0 = np.random.choice([0, 0, 0, 0, 1e-3, 1e-4, 1e-4])
    sde_type = np.random.choice(['ve', 've', 've', 'vp', 'vp', 'simple'])
    num_itr = np.random.choice([250, 500])
    stage = np.random.choice([12, 18])

    output_folder = f'{name}_lr_{lr}_gamma_{lr_gamma}_t0_{t0}_sde_{sde_type}_itr_{num_itr}_stage_{stage}'
    running_comment = f'python main.py --problem-name {name} --forward-net toy  --backward-net toy --dir {output_folder} --log-tb --gpu {gpu}'
    if sde_type == 've':
        sigma_min = np.random.choice([0.03, 0.1, 0.3])
        sigma_max = np.random.choice([0.3, 0.6, 1, 3, 6])
        running_comment += f' --sigma-min {sigma_min} --sigma-max {sigma_max}'
    elif sde_type == 'vp':
        beta_min = np.random.choice([0.03, 0.1, 0.3])
        beta_max = np.random.choice([0.3, 0.6, 1, 3, 6])
        running_comment += f' --beta-min {beta_min} --beta-max {beta_max}'

    os.system(running_comment)
