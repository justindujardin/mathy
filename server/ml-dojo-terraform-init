#!/usr/bin/env bash
set -e


echo "Mount latest model checkpoint from GCS bucket"
gcs_bucket=mzc
sudo mkdir -p /mnt/gcs/mzc
sudo chmod -R a+rwx /mnt/gcs
sudo -u insecurity gcsfuse --dir-mode 777 --file-mode 777 mzc /mnt/gcs/mzc

echo "Configuring nVidia GPU"
sudo nvidia-smi -pm 1

echo "Installing Tensorflow GPU"
sudo /etc/pyenv/bin/pip --no-cache-dir install https://github.com/mind/wheels/releases/download/tf1.7-gpu-nomkl/tensorflow-1.7.0-cp35-cp35m-linux_x86_64.whl
echo "Done."