ARG PYTORCH="1.6.0"
ARG CUDA="10.1"
ARG CUDNN="7"
ARG MM="2.8.0"

FROM nvidia/cuda:10.1-cudnn7-devel-rockylinux8 

FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

# To fix GPG key error when running apt-get update
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/3bf863cc.pub
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/machine-learning/repos/ubuntu1804/x86_64/7fa2af80.pub

RUN apt-get update && apt-get install -y wget \
    && apt-get install -y ffmpeg libsm6 libxext6 git ninja-build libglib2.0-0 libsm6 libxrender-dev libxext6 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# install libraries
RUN pip install -U openmim
RUN mim install mmengine

RUN mim install "mmcv>=1.3.17, <2.0.0"
RUN mim install "mmdet>=2.19.1, <3.0.0"

RUN conda clean --all
ENV FORCE_CUDA="1"

RUN git clone https://github.com/haydenshimada/DNP-mmtracking /mmtracking
WORKDIR /mmtracking
RUN pip install -r requirements/build.txt
RUN pip install -v -e .

# Download dataset
RUN pip install --upgrade --no-cache-dir gdown
RUN mkdir -p /mmtracking/data && mkdir -p /mmtracking/data/DNP 
WORKDIR /mmtracking/data/DNP
# video
RUN gdown --folder https://drive.google.com/drive/folders/11fp02Dx3auXKb1u4bqVVXb2ddEhKF2Un?usp
# gt-reid
# RUN gdown --folder https://drive.google.com/drive/folders/13WseUQKqw5F9VwzQ7Th4dYSzFsycykAP
WORKDIR /mmtracking

# Download checkpoint
RUN mkdir -p /mmtracking/checkpoints && mkdir -p /mmtracking/checkpoints/bytetrack
RUN wget -c https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth -P ./checkpoints/bytetrack

# # Run code
# RUN python notebooks/tracker.py
# RUN pyhon notebooks/toTXT.py

# RUN git commit -m "Run with docker"
# RUN git push origin master
