# build docker image: docker build -f Dockerfile -t mzy22/monodepth:v1.0 .
# create docker container with image: docker run --name=mzy22_monodepth -e COLUMNS=300 --mount type=bind,source="$(pwd)",target=/monodepth_dev -it --gpus all mzy22/monodepth:v1.0

# ubuntu 20.04/cuda
FROM nvidia/cuda:11.4.2-devel-ubuntu20.04
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
         libboost1.71-dev  -y

# 2) set up python module's build environment
RUN apt install --no-install-recommends \
        libpython3-dev \
        python3-pip -y

# 3) set up python run environment
# Put everything in some subfolder
WORKDIR "/monodepth"
COPY code/ ./
RUN mkdir data/ && mkdir data/erp_00/
COPY data/erp_00/0001_rgb.jpg ./data/erp_00/

RUN pip3 install -r ./python/requirements.txt

#-- build python cpp module
# 1) build the cpp project
RUN cd cpp && mkdir build && cd build && cmake ..  -DCMAKE_BUILD_TYPE=Release && make -j

# 2) build & install python module
RUN cd cpp/python/ && python3 ./setup.py build && python3 ./setup.py bdist_wheel && pip3 install dist/instaOmniDepth-0.1.0-cp38-cp38-linux_x86_64.whl

# 3) run test script
RUN cd /monodepth/python/src/test && python3 ./test_depthmapAlign_module.py --task 1
