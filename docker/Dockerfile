# Official tensorflow images built after Sept 2021 are based on Ubuntu 20.04
# Check: https://hub.docker.com/r/tensorflow/tensorflow/

FROM tensorflow/tensorflow:latest-gpu AS BASE_BUILD

FROM BASE_BUILD AS BUILD1
RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip && \
        apt-get -y install vim && \
        pip install tflearn && \
        pip install opencv-python && \
        apt-get install ffmpeg libsm6 libxext6 -y

FROM BUILD1 AS BUILD2
#COPY . /home/gpgtpu
WORKDIR /home

# use python convention
RUN ln -s $(which python3) /usr/bin/python

#FROM BUILD2 AS BUILD3
# Installs the latest version by default.
#ARG TF_PACKAGE=tensorflow-gpu
#ARG TF_PACKAGE_VERSION=
#RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

