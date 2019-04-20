#!/usr/bin/env bash
set +e

echo "Got shutdown signal! Copying a file for shits."
touch /mnt/gcs/mzc/shutdown_signal.txt
cp ~/mathy/nohup.out /mnt/gcs/mzc/shutdown_nohup.txt
