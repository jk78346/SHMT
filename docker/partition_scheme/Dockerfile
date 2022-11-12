#FROM nvcr.io/nvidia/l4t-ml:r32.5.0-py3 AS BASIC_BUILD
FROM ubuntu:18.04 AS BASIC_BUILD
RUN ln -snf /usr/share/zoneinfo/$CONTAINER_TIMEZONE /etc/localtime && echo $CONTAINER_TIMEZONE > /etc/timezone && \
    apt-get update && \
    apt-get install -y vim cmake g++

# install opencv
FROM BASIC_BUILD AS BUILD1
RUN apt-get update && \
    apt-get install -y build-essential apt-utils && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq libopencv-dev && \
    apt-get install -y libatlas-base-dev

# install libedgetpu, reference: google-coral/libedgetpu/docker/Dockerfile
FROM BUILD1 AS BUILD2
COPY update_sources.sh /
RUN /update_sources.sh
RUN dpkg --add-architecture armhf
RUN dpkg --add-architecture arm64
 
#RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
#    libc6-dev:arm64 
#    libc6-dev:armhf 
RUN DEBIAN_FRONTEND=noninteractive apt-get install -y \
    python3-all \
    python3-numpy \
    build-essential \ 
#    crossbuild-essential-armhf \
#    crossbuild-essential-arm64 \
    libusb-1.0-0-dev \
#    libusb-1.0-0-dev:arm64 \
#    libusb-1.0-0-dev:armhf \
    zlib1g-dev \
#    zlib1g-dev:armhf \
#    zlib1g-dev:arm64 \
    sudo \
    debhelper \
    pkg-config \
    zip \
    unzip \
    curl \
    wget \
    git \
    tree \
    software-properties-common \
    $(grep Ubuntu /etc/os-release > /dev/null && echo vim-common || echo xxd)

RUN apt-get install -y libeigen3-dev && \
    sudo ln -s /usr/include/engin3/Eigen /usr/include/Eigen
# for quick scipy.signal.fftconvolve test in python2 only
RUN apt-get install -y python-scipy 

FROM BUILD2 AS FINAL 
# setup LD_LIBRARY_PATH
RUN ln -s $(which python3) /usr/bin/python && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/libedgetpu/out/direct/k8" >> ~/.bashrc && \
    echo "export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/home/libedgetpu/out/direct/aarch64" >> ~/.bashrc
WORKDIR /home

