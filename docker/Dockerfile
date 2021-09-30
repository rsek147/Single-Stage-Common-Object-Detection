ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

SHELL ["/bin/bash", "-c"]

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get upgrade -y && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
 git libgl1-mesa-dev curl htop wget ca-certificates \
 && apt-get clean \
 && rm -rf /var/lib/apt/lists/*

 # Install Miniconda
RUN curl -LO https://repo.continuum.io/miniconda/Miniconda3-latest-Linux-x86_64.sh \
 && bash Miniconda3-latest-Linux-x86_64.sh -p /miniconda -b \
 && rm Miniconda3-latest-Linux-x86_64.sh
ENV PATH=/miniconda/bin:$PATH

WORKDIR /work
RUN git clone https://github.com/rsek147/Single-Stage-Common-Object-Detection.git sscod
WORKDIR /work/sscod

RUN conda init bash
RUN conda install -y conda-build \
 && conda env create -f environment.yml \
 && conda clean -ya

ENV FORCE_CUDA="1"
RUN /miniconda/envs/sscod/bin/pip install --upgrade pip \
 && /miniconda/envs/sscod/bin/pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI" \
 && ln -s /usr/local/cuda-10.2/targets/x86_64-linux/include/cublas_v2.h /usr/local/cuda-10.1/targets/x86_64-linux/include/ \
 && ln -s /usr/local/cuda-10.2/targets/x86_64-linux/include/cublas_api.h /usr/local/cuda-10.1/targets/x86_64-linux/include/ \
 && /miniconda/envs/sscod/bin/pip install -v -e .
