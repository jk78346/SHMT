#!/bin/sh

PROJ=gpgtpu
DATASET_DIR=/nfshome/khsu037/ILSVRC
TARGET_DIR=/mnt # the dataset mount point within container
IMAGE_NAME=${PROJ}_image
CONTAINER_NAME=${PROJ}_container
DOCKERFILE_PATH=./../samples/0_Simple/GPGTPU_conv_blocking

# build dockerfile to generate docker image
echo "[${PROJ}] - building docker image from dockerfile..."
sudo docker build -t ${IMAGE_NAME} ${DOCKERFILE_PATH}

# generate container from image
docker stop ${CONTAINER_NAME}
docker rm ${CONTAINER_NAME}

# mount dataset dir (ImageNet) from host fs to container fs
echo "[${PROJ}] - build docker container and mouting dataset dir to container..."
sudo docker run -d -it --name ${CONTAINER_NAME} --mount type=bind,source=${DATASET_DIR},target=${TARGET_DIR} ${IMAGE_NAME} bash

