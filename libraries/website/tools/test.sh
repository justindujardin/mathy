#!/bin/bash
set -e
echo "Activating virtualenv... (if this fails you may need to run setup.sh first)"
. .env/bin/activate
echo "Running tests..."
python -m pytest --cov=mathy -m "not mptest"
# NOTE: we run the multiprocessing "mp" test separately because Tensorflow 
#       has to be imported from a subprocess first for it to work at all.
#       these tests are marked like so:
# 
# @pytest.mark.mptest
# def test_ml_zero_training():
#     from ...docs.snippets.ml import zero_training
python -m pytest --cov=mathy --cov-append -m "mptest"
