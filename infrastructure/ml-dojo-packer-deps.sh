#!/usr/bin/env bash
set -e

sudo apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1704/x86_64/7fa2af80.pub
sudo apt-get update
echo "Installing environment basics..."
sudo apt-get install -y build-essential
sudo apt-get install -y unzip libssl-dev zlib1g-dev libbz2-dev \
    libreadline-dev wget curl \
    xz-utils python-pip python-virtualenv python3-pip python3-venv \
    python-dev python3-dev google-cloud-sdk

echo "Downloading and installing CUDA repo packages..."
gsutil cp gs://shm-builds/cuda-repo-ubuntu1704_9.0.176-1_amd64.deb /tmp/
sudo dpkg -i /tmp/cuda-repo-ubuntu1704_9.0.176-1_amd64.deb

echo "Downloading and installing CUDA packages..."
gsutil cp gs://shm-builds/cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb /tmp/
sudo dpkg -i /tmp/cuda-repo-ubuntu1704-9-0-local_9.0.176-1_amd64.deb

echo "Downloading and installing CUDA packages (Update 1)..."
gsutil cp gs://shm-builds/cuda-repo-ubuntu1704-9-0-local-cublas-performance-update_1.0-1_amd64.deb /tmp/
sudo dpkg -i /tmp/cuda-repo-ubuntu1704-9-0-local-cublas-performance-update_1.0-1_amd64.deb

echo "Downloading and installing CUDA packages (Update 2)..."

gsutil cp gs://shm-builds/cuda-repo-ubuntu1704-9-0-local-cublas-performance-update-2_1.0-1_amd64.deb /tmp/
sudo dpkg -i /tmp/cuda-repo-ubuntu1704-9-0-local-cublas-performance-update-2_1.0-1_amd64.deb

echo "Downloading and installing CUDA packages (Update 3)..."
gsutil cp gs://shm-builds/cuda-repo-ubuntu1704-9-0-local-cublas-performance-update-3_1.0-1_amd64.deb /tmp/
sudo dpkg -i /tmp/cuda-repo-ubuntu1704-9-0-local-cublas-performance-update-3_1.0-1_amd64.deb

echo "Downloading and installing CUDA Neural Net library..."
gsutil cp gs://shm-builds/libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb /tmp/
sudo dpkg -i /tmp/libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb

echo "Installing and updating pip..."
sudo pip3 install --upgrade pip 
sudo pip3 install --upgrade virtualenv

echo "Create virtualenv at /etc/pyenv/"
sudo mkdir -p /etc/pyenv/
sudo python3 -m venv /etc/pyenv/
