#!/bin/sh

GITTOP="$(git rev-parse --show-toplevel 2>&1)"

#source configure file
. "${GITTOP}/configure.cfg"
PROJ=${GPGTPU_PARTITION}

CONTAINER_NAME=${PROJ}_container
# use "-u $(id -u):$(id -g)" to login the docker container with a tmp acount ID that is not sudo user.
docker exec -it ${CONTAINER_NAME} bash
