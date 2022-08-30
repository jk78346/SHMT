#!/bin/sh

PROJ=gpgtpu
DOCKERFILE_PATH=./../docker
IMAGE_NAME=${PROJ}_image
CONTAINER_NAME=${PROJ}_container
DATASET_DIR=/nfshome/khsu037/ILSVRC
SRC_DIR=~/GPGTPU/

DATASET_TARGET_DIR=/mnt # the dataset mount point within container
SRC_TARGET_DIR=/home/ # the src code mount point within container

# build dockerfile to generate docker image
echo "[${PROJ}] - building docker image from dockerfile..."
sudo docker build -t ${IMAGE_NAME} ${DOCKERFILE_PATH}

docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

# generate container from image
# mount dataset dir (ImageNet)/src  from host fs to container fs
# get the container running
echo "[${PROJ}] - build docker container..."
sudo docker run -d \
         -it \
         --gpus all \
         --name ${CONTAINER_NAME} \
         --mount type=bind,source=${DATASET_DIR},target=${DATASET_TARGET_DIR} \
         --mount type=bind,source=${SRC_DIR},target=${SRC_TARGET_DIR} \
         ${IMAGE_NAME} \
         bash

