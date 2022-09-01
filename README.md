# GPGTPU
General-Purposed GPU and TPU project (Jetson nano + edgeTPU)

# Pre-requisites
1. NVIDIA Drivers

# Setup
## 1. Install docker
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

## 2. Install NVIDIA Container Toolkit 
Please refer to: https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html#docker



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


# Trouble shooting
1. ```docker: Error response from daemon: could not select device driver "" with capabilities: [[gpu]].``` \
Follow this steps to uninstall and install docker for nvidia image: https://github.com/NVIDIA/nvidia-docker/issues/1637#issuecomment-1130151618
