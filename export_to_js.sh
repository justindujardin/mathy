#!/bin/bash

tensorflowjs_converter \
    --input_format=tf_session_bundle \
    --output_node_names='out_policy,out_value' \
    ./training/web_1 \
    ./training/web_js_1
