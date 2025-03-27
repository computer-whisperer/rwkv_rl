FROM nvidia/cuda:12.6.3-cudnn-devel-ubuntu24.04

WORKDIR /usr/src/mavis

RUN mkdir -p /ceph-fuse/public/neural_models/


RUN apt update && apt install -y \
    build-essential \
    libxext6 \
    libegl1 \
    python3 \
    cmake \
    openssl \
    pkg-config \
    libssl-dev \
    openssh-server \
    sudo \
    git \
    rustup \
    libvulkan1 \
    libvulkan-dev \
    vulkan-tools \
    libnvidia-gl-565 \
    protobuf-compiler \
    libcutlass-dev \
    python3-dev \
    python3-pip \
    python3-venv \
RUN pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu126
RUN rustup default nightly

RUN useradd -rm -d /home/ubuntu -s /bin/bash -g root -G sudo -u 1001 test

RUN echo 'test:test' | chpasswd
RUN echo 'ubuntu:ubuntu' | chpasswd

RUN service ssh start

RUN echo "export PATH=\"/usr/local/nvidia/bin:/usr/local/cuda/bin:$PATH\"" >> /home/ubuntu/.bashrc

COPY sshd_config /etc/ssh/

COPY . .
RUN cargo build

USER ubuntu
RUN rustup default nightly
USER root

COPY nvidia_icd.json /etc/vulkan/icd.d