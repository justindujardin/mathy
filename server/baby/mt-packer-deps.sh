#!/usr/bin/env bash
set -e

echo "Installing environment basics..."
sudo apt-get update
sudo apt-get install -y build-essential
sudo apt-get install -y unzip libssl-dev zlib1g-dev libbz2-dev libreadline-dev wget curl xz-utils python3-pip python3-venv python3-dev google-cloud-sdk

echo "Installing GCS Fuse for model sync"
export GCSFUSE_REPO=gcsfuse-`lsb_release -c -s`
echo "deb http://packages.cloud.google.com/apt $GCSFUSE_REPO main" | sudo tee /etc/apt/sources.list.d/gcsfuse.list
curl https://packages.cloud.google.com/apt/doc/apt-key.gpg | sudo apt-key add -
sudo apt-get update
sudo apt-get install -y gcsfuse

echo "Downloading and installing CUDA..."
curl -O https://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-key adv --fetch-keys http://developer.download.nvidia.com/compute/cuda/repos/ubuntu1804/x86_64/7fa2af80.pub
sudo dpkg -i cuda-repo-ubuntu1804_10.0.130-1_amd64.deb
sudo apt-get update
sudo apt-get install -y cuda-10-0

echo "Downloading and installing CUDA Neural Net library..."
gsutil cp gs://shm-builds/libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb /tmp/
sudo dpkg -i /tmp/libcudnn7_7.4.2.24-1+cuda10.0_amd64.deb

