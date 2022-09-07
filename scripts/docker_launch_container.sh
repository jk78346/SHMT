CONTAINER_NAME=gpgtpu_container
# use "-u $(id -u):$(id -g)" to login the docker container with a tmp acount ID that is not sudo user.
sudo docker exec -it -u $(id -u):$(id -g) ${CONTAINER_NAME} bash
