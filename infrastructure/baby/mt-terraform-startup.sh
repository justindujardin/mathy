#!/usr/bin/env bash
set -e

echo "Mount latest model checkpoint from GCS bucket"
gcs_bucket=mzc
sudo mkdir -p /mnt/gcs/mzc
sudo chmod -R a+rwx /mnt/gcs
sudo -u insecurity gcsfuse --dir-mode 777 --file-mode 777 mzc /mnt/gcs/mzc

echo "Configuring nVidia GPU"
sudo nvidia-smi -pm 1


# 
# Write out a startup script for hte user when they log in
# 
sudo touch /usr/local/bin/dev-init
(sudo -u insecurity cat <<-'EOF'
#!/bin/bash
set -e
echo "Create python3.6 virtualenv and activate it..."
echo "Setting up \$PATH..."
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc

echo "Installing Tensorflow GPU"
pip3 --no-cache-dir install tf-nightly-gpu
echo "Done."

echo "Installing SSH keys from bucket..."
gsutil cp gs://shm-secrets/id_rsa ~/.ssh/id_rsa
chmod 600 ~/.ssh/id_rsa
gsutil cp gs://shm-secrets/id_rsa.pub ~/.ssh/id_rsa.pub

git config --global user.email "justin@dujardinconsulting.com"
git config --global user.name "Justin DuJardin"

echo "Cloning mz..."
git clone git@github.com:justindujardin/mathy.git

echo "Installing requirements..."
cd mathy
pip3 install -r requirements.txt

echo "Done."

EOF
) | tee /home/insecurity/dev-init
chmod +x /home/insecurity/dev-init

# 
# Write out a spacy pretraining startup script
# 
sudo touch /usr/local/bin/spacy-init
(sudo -u insecurity cat <<-'EOF'
#!/bin/bash
set -e
echo "Create python3.6 virtualenv and activate it..."
echo "Setting up \$PATH..."
echo 'export CUDA_HOME=/usr/local/cuda' >> ~/.bashrc
echo 'export PATH=$PATH:$CUDA_HOME/bin' >> ~/.bashrc
echo 'export LD_LIBRARY_PATH=/usr/local/cuda/extras/CUPTI/lib64:$LD_LIBRARY_PATH' >> ~/.bashrc
source ~/.bashrc
python3.6 -m spacy download en_core_web_lg
echo "Pretrain command: nohup python3.6 -m spacy pretrain /mnt/gcs/mzc/spacy_pretraining/toxic_pretraining_2735618_tweets.jsonl en_core_web_lg /mnt/gcs/mzc/spacy_pretraining/toxic_n2735618/ &"

EOF
) | tee /home/insecurity/spacy-init
chmod +x /home/insecurity/spacy-init
