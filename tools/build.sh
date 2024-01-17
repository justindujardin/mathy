#!/bin/bash
set -e
echo "Build python package..."
.env/bin/python setup.py sdist bdist_wheel
