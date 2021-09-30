ARG PYTORCH="1.3"
ARG CUDA="10.1"
ARG CUDNN="7"

# FROM pytorch/pytorch:${PYTORCH}-cuda${CUDA}-cudnn${CUDNN}-devel
FROM nvidia/cuda:${CUDA}-cudnn${CUDNN}-devel-ubuntu18.04

RUN echo 'debconf debconf/frontend select Noninteractive' | debconf-set-selections

ENV TORCH_CUDA_ARCH_LIST="6.0 6.1 7.0+PTX"
ENV TORCH_NVCC_FLAGS="-Xfatbin -compress-all"
ENV CMAKE_PREFIX_PATH="$(dirname $(which conda))/../"

RUN apt-get update && apt-get upgrade && apt-get install -y libglib2.0-0 libsm6 libxrender-dev libxext6 \
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

RUN conda init bash \
 && conda install -y conda-build \
 && conda create -f environment.yml \
 && conda activate sscod \
 && conda clean -ya

RUN pip install --upgrade pip \
 && pip install "git+https://github.com/cocodataset/cocoapi.git#subdirectory=PythonAPI" \
 && pip install -v -e .

# ENV CONDA_DEFAULT_ENV=sscod
# ENV CONDA_PREFIX=/miniconda/envs/$CONDA_DEFAULT_ENV
# ENV PATH=$CONDA_PREFIX/bin:$PATH
# ENV CONDA_AUTO_UPDATE_CONDA=false

# # Install mmdetection
# RUN conda clean --all
# RUN git clone https://github.com/open-mmlab/mmdetection.git /mmdetection
# WORKDIR /mmdetection
# ENV FORCE_CUDA="1"
# RUN pip install --no-cache-dir -e .
