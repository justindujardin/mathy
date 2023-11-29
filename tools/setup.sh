#!/bin/bash
set -e

# Default Python path
PYTHON_PATH="python3"

# Check if a custom Python path is provided as the first argument
if [ -n "$1" ]; then
  PYTHON_PATH="$1"
fi

echo "Using Python at: $PYTHON_PATH"
$PYTHON_PATH --version

# Make the virtualenv only if the folder doesn't exist
DIR=.env
if [ ! -d "${DIR}" ]; then
  echo "Installing root virtualenv (.env)"
  pip install virtualenv --upgrade
  # The first syntax is for CI (travis) and the OR is for MacOS catalina
  $PYTHON_PATH -m virtualenv -p $PYTHON_PATH .env || virtualenv -p $PYTHON_PATH .env
fi
echo "Installing/updating requirements..."
.env/bin/pip install -e ./libraries/mathy_python

sh tools/ci-setup.sh
