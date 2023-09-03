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

"""
checkerboard best settings:
   best vpsde: 500itr, stage=12, beta=0.03, beta=0.6, lr=6e-4, gamma=0.8, t0=1e-4
   minor   ve: 500itr, stage=18, sigma=0.03, max=0.3, lr=3e-4, gamma=0.9, t0=0  

GMM best settings:
        VPSDE: 500itr, stage=18, beta=0.03, 0.6, lr=1e-3, gamma=0.8, t0=0

Moon-to-Spiral
        VPSDE: 250itr, stage=18, beta=0.03, max=3, lr=6e-4, gamma=0.8, t0=0

"""


for i in range(3):
    name = np.random.choice(['checkerboard'])
    lr = np.random.choice([6e-4, 4e-4, 8e-4])
    lr_gamma = np.random.choice([0.8])
    t0 = np.random.choice([0, 3e-4, 1e-4, 3e-5])
    sde_type = np.random.choice(['vp'])
    num_itr = np.random.choice([400, 500, 800])
    stage = np.random.choice([12, 18, 24])

    output_folder = f'{name}_lr_{lr}_gamma_{lr_gamma}_t0_{t0}_sde_{sde_type}_itr_{num_itr}_stage_{stage}'
    running_comment = f'python main.py --problem-name {name} --forward-net toy  --backward-net toy --log-tb --gpu {gpu}'
    if sde_type == 've':
        sigma_min = np.random.choice([0.03, 0.1, 0.3])
        sigma_max = np.random.choice([0.3, 0.6, 1, 3, 6])
        running_comment += f' --sigma-min {sigma_min} --sigma-max {sigma_max}'
        output_folder += f'_sigma_min_{sigma_min}_max_{sigma_max}'
    elif sde_type == 'vp':
        beta_min = np.random.choice([0.02, 0.03, 0.04, 0.05])
        beta_max = np.random.choice([0.4, 0.5, 0.6, 0.7, 0.8])
        running_comment += f' --beta-min {beta_min} --beta-max {beta_max}'
        output_folder += f'_beta_min_{beta_min}_max_{beta_max}'
    running_comment += f' --lr {lr} --lr-gamma {lr_gamma} --t0 {t0} --sde-type {sde_type} --num-itr {num_itr} --num-stage {stage} --dir {output_folder}'

    os.system(running_comment)
