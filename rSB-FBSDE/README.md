# Reflected Schrödinger Bridge using Reflected Forward-Backward SDEs with Robin and Neumann boundary conditions


## Examples

| p0 ⇆ pT (`--problem-name`)  | Results (blue/left: p0 ← pT, red/right: p0 → pT) |
|-------------------------|-------------------------|
| Mixture Gaussians ⇆ Gaussian (`gmm`) | <img src="./assets/gmm.png" alt="drawing" width="400"/> |
| CheckerBoard ⇆ Gaussian (`checkerboard`) | <img src="./assets/checkerboard.png" alt="drawing" width="400"/> | 
| Spiral ⇆ Moon (`moon-to-spiral`) | <img src="./assets/spiral.png" alt="drawing" width="400"/> | 
| CIFAR-10 ⇆ Gaussian (`cifar10`) | <p float="left"> <img src="./assets/cifar10-forward.gif" alt="drawing" width="180"/>  <img src="./assets/cifar10-backward.gif" alt="drawing" width="180"/> </p> |

## Installation

This code is developed with Python3. PyTorch >=1.7 (we recommend 1.8.1). First, install the dependencies with [Anaconda](https://www.anaconda.com/products/individual) and activate the environment `sb-fbsde` with
```bash
conda env create --file requirements.yaml python=3
conda activate sb-fbsde
```

## Training

```bash
python main.py --problem-name moon-to-spiral --forward-net toy  --backward-net toy --dir output --log-tb --ckpt-freq 2
python main.py --problem-name smile-to-checkerboard --forward-net toy  --backward-net toy --dir output --log-tb
python main.py --problem-name gmm --forward-net toy  --backward-net toy --dir output --log-tb --ckpt-freq 2
```

##  Acknowledge

This repo heavily depends on [link](https://github.com/ghliu/SB-FBSDE)
