FROM ubuntu:18.04 AS BASIC_BUILD
RUN apt-get update && \
    apt-get install -y vim cmake g++ git 

#install opencv
FROM BASIC_BUILD AS BUILD1
RUN apt-get update && \
    apt-get install -y build-essential apt-utils && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq libopencv-dev

WORKDIR /home
