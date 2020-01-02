#!/bin/bash
set -e
. .env/bin/activate

echo "Replacing package readme with root..."
cp ../../README.md ./

echo "Build python package..."
python setup.py sdist bdist_wheel
