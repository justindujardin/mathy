#!/bin/bash
set -e

. ../../.env/bin/activate

python -m tools.docs
