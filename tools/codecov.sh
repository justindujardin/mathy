#!/bin/bash
set -e
. ./libraries/website/.env/bin/activate
pip install codecov
python -m codecov
