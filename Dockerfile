# build docker image: docker build -f Dockerfile -t mzy22/monodepth:v1.0 .
# create docker container with image: docker run --name=mzy22_monodepth -e COLUMNS=300 --mount type=bind,source="$(pwd)",target=/monodepth_dev -it --gpus all mzy22/monodepth:v1.0

# ubuntu 20.04/cuda
FROM nvidia/cuda:11.4.3-devel-ubuntu20.04
# FROM alpine:3.4

#-- setup building environment 
RUN apt-get update

# 1) set up cpp code building dependent 3rd party libraries
ARG DEBIAN_FRONTEND=noninteractive
ENV TZ=Europe/London

RUN apt-get install --no-install-recommends \
        build-essential \
         cmake \
         libeigen3-dev \
         libgoogle-glog-dev \
         libgtest-dev \
         libopencv-dev \
         libceres-dev \
         python3-pybind11 \
         git \
         wget \
         libboost1.71-dev \
         libc6 -y

# 2) set up python module's build environment
RUN apt install --no-install-recommends \
        libpython3-dev \
        python3-pip -y

# 3) set up python run environment
# Put everything in some subfolder
WORKDIR "/monodepth"
COPY . ./

# Setup BoostingMonocularDepth
RUN git submodule update --init
RUN cd ./BoostingMonocularDepth/pix2pix/ && mkdir -p checkpoints/mergemodel
# Midas weights
RUN wget https://github.com/isl-org/MiDaS/releases/download/v2_1/model-f6b98070.pt -O ./BoostingMonocularDepth/midas/model.pt
# Merge net weights
RUN wget -P ./BoostingMonocularDepth/pix2pix/checkpoints/mergemodel https://www.sfu.ca/~yagiz/CVPR21/latest_net_G.pth

RUN pip3 install -r ./code/python/requirements.txt

#-- build python cpp module
# 1) build the cpp project
RUN cd ./code/cpp && mkdir build && cd build && cmake ..  -DCMAKE_BUILD_TYPE=Release && make -j

# 2) build & install python module
RUN cd ./code/cpp/python/ && python3 ./setup.py build && python3 ./setup.py bdist_wheel && pip3 install dist/instaOmniDepth-0.1.0-cp38-cp38-linux_x86_64.whl

# Download necessary models by running test_experiment
RUN cd /monodepth/code/python/src && python3 main.py --expname test_experiment --blending_method all --grid_size 8x7
