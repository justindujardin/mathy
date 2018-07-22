#!/usr/bin/env bash
set -e

echo "Updating system packages and installing python/gcloud..."
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y unzip libssl-dev libcupti-dev zlib1g-dev libbz2-dev \
    libreadline-dev wget curl \
    xz-utils python3-pip python3-venv python3-dev google-cloud-sdk

echo "Installing and updating pip..."
sudo pip3 install --upgrade pip 
sudo pip3 install --upgrade virtualenv

echo "Installing CUDA 9 for ubuntu 1604..."
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/7fa2af80.pub
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1604/x86_64/cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu1604_9.0.176-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda-9-0
echo "Downloading and installing CUDA Neural Net library..."
gsutil cp gs://shm-builds/libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb /tmp/
sudo dpkg -i /tmp/libcudnn7_7.1.4.18-1+cuda9.0_amd64.deb

echo "Setting up \$PATH..."
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "Installing Tensorflow GPU"
sudo pip3 install tensorflow-gpu

echo "Installing Keras and PyTorch"
sudo pip3 install keras spacy torch torch-vision
