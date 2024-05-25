
# CONDA Virtual Enviorment 

## Miniconda

reference: https://docs.anaconda.com/free/miniconda/
```
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm -rf ~/miniconda3/miniconda.sh
```

After installing, initialize your newly-installed Miniconda. The following commands initialize for bash and zsh shells:

```
~/miniconda3/bin/conda init bash
~/miniconda3/bin/conda init zsh
```

restart session

## Env file refernece for working torch cuda env


```
WorkshopCUDAEnv.yml

name: WorkshopCUDAEnv
channels:
  - pytorch
  - nvidia
  - conda-forge
dependencies:
  - python=3.11
  - pytorch 
  - scipy
  - torchvision 
  - torchaudio 
  - pytorch-cuda=12.1
  - cupy
  ```

## Create Env

`conda env create --file WorkshopCUDAEnv.yml`

## Activate Env

`conda activate WorkshopCUDAEnv`

## Check CUDA enabled

inside our new Virtual Env run `python`

```
import torch
torch.cuda.is_available()
```

you should get `True`

## OpenCV support

```
sudo apt install libopengl0 -y
sudo apt install libegl1 -y
```

inside our new Virtual Env run `python`

```
import cv2
```
