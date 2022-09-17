FROM ubuntu:18.04 AS BASIC_BUILD
RUN apt-get update && \
    apt-get install -y vim cmake g++ git 

#install opencv
FROM BASIC_BUILD AS BUILD1
RUN apt-get update && \
    apt-get install -y build-essential apt-utils && \
    DEBIAN_FRONTEND=noninteractive apt-get install -yq libopencv-dev

# install libedgetpu
FROM BUILD1 AS BUILD2
RUN apt-get update && \
    apt-get install -y python3-pip python-dev pkg-config wget usbutils curl && \
    echo "deb https://packages.cloud.google.com/apt coral-edgetpu-stable main" \
    | tee /etc/apt/sources.list.d/coral-edgetpu.list && \
    curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | apt-key add - && \
    apt-get update
#    git clone https://github.com/google-coral/pycoral.git && \
#    git clone https://github.com/google-coral/test_data.git && \
#    apt install -y python3-pycoral

FROM BUILD2 AS FINAL 
WORKDIR /home
