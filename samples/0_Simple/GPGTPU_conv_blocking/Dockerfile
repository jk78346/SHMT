ARG UBUNTU_VERSION=20.04
FROM ubuntu:${UBUNTU_VERSION}
MAINTAINER khsu037

COPY . /usr/src/gpgtpu
WORKDIR /usr/src/gpgtpu

RUN apt-get update && apt-get install -y \
        python3 \
        python3-pip

# use python convention
RUN ln -s $(which python3) /usr/bin/python

# Installs the latest version by default.
ARG TF_PACKAGE=tensorflow
ARG TF_PACKAGE_VERSION=
RUN python3 -m pip install --no-cache-dir ${TF_PACKAGE}${TF_PACKAGE_VERSION:+==${TF_PACKAGE_VERSION}}

RUN python show_tf_version.py
