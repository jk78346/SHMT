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
        apt-get install ffmpeg libsm6 libxext6 -y && \
        apt-get install time && \
        pip install gitpython && \
        apt-get install -y git

FROM BUILD1 AS BUILD2
RUN apt-get install --no-install-recommends -y \
        gnupg \
        ca-certificates \
        curl \
        apt-utils \
        apt-transport-https
RUN curl https://packages.cloud.google.com/apt/doc/apt-key.gpg \
        | apt-key add -
RUN echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
        | tee /etc/apt/sources.list.d/coral-edgetpu.list
RUN apt-get update && apt-get install -y edgetpu-compiler

FROM BUILD2 AS BUILD3
#COPY . /home/gpgtpu
WORKDIR /home

# use python convention
RUN ln -s $(which python3) /usr/bin/python

#FROM BUILD2 AS BUILD3
# Installs the latest version by default.
#ARG TF_PACKAGE=tensorflow-gpu
#ARG TF_PACKAGE_VERSION=
#RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

