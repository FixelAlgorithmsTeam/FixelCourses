# Connecting SSH

ssh example

`ssh -i "~/.ssh/sagi-validit-app-gpu-us-east-1.pem" ubuntu@ec2-54-196-245-33.compute-1.amazonaws.com`

# Install NVIDIA Driver

check nvidia card
```lspci | grep -i nvidia```

install gcc

```
sudo apt update

sudo apt-get upgrade

sudo apt install build-essential
```

reboot `sudo reboot`

run `nvidia-smi`
you should see something like

```
ubuntu@ip-172-31-26-222:~$ nvidia-smi
Command 'nvidia-smi' not found, but can be installed with:
sudo apt install nvidia-utils-390         # version 390.157-0ubuntu0.22.04.2, or
sudo apt install nvidia-utils-418-server  # version 418.226.00-0ubuntu5~0.22.04.1
sudo apt install nvidia-utils-450-server  # version 450.248.02-0ubuntu0.22.04.1
sudo apt install nvidia-utils-470         # version 470.223.02-0ubuntu0.22.04.1
sudo apt install nvidia-utils-470-server  # version 470.223.02-0ubuntu0.22.04.1
sudo apt install nvidia-utils-525         # version 525.147.05-0ubuntu0.22.04.1
sudo apt install nvidia-utils-525-server  # version 525.147.05-0ubuntu0.22.04.1
sudo apt install nvidia-utils-535         # version 535.129.03-0ubuntu0.22.04.1
sudo apt install nvidia-utils-535-server  # version 535.129.03-0ubuntu0.22.04.1
sudo apt install nvidia-utils-510         # version 510.60.02-0ubuntu1
sudo apt install nvidia-utils-510-server  # version 510.47.03-0ubuntu3
```

install drivers

```
sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers install

```

reboot `sudo reboot`

run `nvidia-smi`

you should get something like

```
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.161.07             Driver Version: 535.161.07   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:00:1E.0 Off |                    0 |
| N/A   31C    P8               9W /  70W |      2MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|  No running processes found                                                           |
+---------------------------------------------------------------------------------------+
```
