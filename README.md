# GPGTPU
General-Purposed GPU and TPU project (Jetson nano + edgeTPU). \
This project contains two major parts: kernel model training scheme and the acutal partition execution scheme. These two schemes require two different cooresponding platforms. 

## Tested platforms
The training mode: Ubuntu 20.04 x86_64 12th Gen Intel(R) Core(TM) i9-12900KF (Rayquaza in Escal's environment)
The partition execution mode: Ubuntu 18.04 aarch64 Cortex-A57 (nano-2 in Escal's environment)


## Pre-requisites
1. NVIDIA Drivers

## Setup
### 1. Install docker
Please refer to: https://www.simplilearn.com/tutorials/docker-tutorial/how-to-install-docker-on-ubuntu \
or the official website: https://docs.docker.com/engine/install/ubuntu/

```
sudo apt-get remove docker docker-engine docker.io
sudo apt-get update
sudo apt install docker.io
sudo snap install docker
```
Check docker verison
```
docker --version
```
Run docker hello-world
```
sudo docker run hello-world
```

### 2. Install NVIDIA Container Toolkit 
Please refer to: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker

1. install ```curl```
```
sudo apt update && sudo apt upgrade
sudo apt install curl
```
2. install the NVIDIA Container Toolkit
``` 
distribution=$(. /etc/os-release;echo $ID$VERSION_ID) \
&& curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg \
&& curl -s -L https://nvidia.github.io/libnvidia-container/$distribution/libnvidia-container.list | \
   sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
   sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
sudo apt-get update
sudo apt-get install -y nvidia-docker2
sudo systemctl restart docker
```

## Build and execution
### 1. The kernel model training mode (on Rayquaza as example)
(TBD)
### 2. The partition execution mode (on Jetson nano-2 as example)
To build project and run a single CPU v.s. GPU example:
```
(on host at $GITTOP)$ mkdir -p nano_host_build && cd nano_host_build
(on host at $GITTOP)$ cmake .. && make
(on host at $GITTOP)$ sudo ./gpgtpu sobel_2d 1024 1 cpu gpu
```

To build project and run a single CPU v.s. edgeTPU example:
```
(on host at $GITTOP)$ sh scripts/docker_setup_partition.sh
(on host at $GITTOP)$ sh scripts/docker_launch_partition.sh
(in docker at $GITTOP)$ sh scripts/nano_docker_unit_test.sh
```


## Trouble shooting
### 1. ```docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].``` 
1. Follow this steps to uninstall and install docker for nvidia image: https://github.com/NVIDIA/nvidia-docker/issues/1637#issuecomment-1130151618. 
2. Make sure the following command gives good ```nvidia-smi``` output: \
```sudo docker run --rm --gpus all nvidia/cuda:11.7.0-devel-ubuntu20.04 nvidia-smi``` \
(Replace version numbers accordingly if cuda and Ubuntu versions vary.)

### 2. ```scripts/docker_setup.sh: 7: .: Can't open fatal: unsafe repository ('/home/kuanchiehhsu/GPGTPU' is owned by someone else)```
To add an exception for this directory, call:
```
git config --global --add safe.directory '*'
```
reference: https://stackoverflow.com/questions/71901632/fatal-error-unsafe-repository-home-repon-is-owned-by-someone-else
