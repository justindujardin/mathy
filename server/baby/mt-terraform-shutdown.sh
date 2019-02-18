#!/usr/bin/env bash
set +e

echo "Got shutdown signal! Copying a file for shits."
touch /mnt/gcs/mzc/shutdown_signal.txt
cp ~/mathzero/nohup.out /mnt/gcs/mzc/shutdown_nohup.txt
