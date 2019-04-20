#!/bin/bash

tensorflowjs_converter \
    --input_format=tf_session_bundle \
    --output_node_names='out_policy,out_value' \
    /mnt/gcs/mzc/love/train \
    ./exports/mathy_alpha
