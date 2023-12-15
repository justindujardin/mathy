#!/bin/bash
set -e

# Make the virtualenv only if the folder doesn't exist
DIR=../../.env
if [ ! -d "${DIR}" ]; then
  echo "Root env is not found. Run setup.sh from the project root first."
  exit 1
fi

# Use root env
. ../../.env/bin/activate
echo "Installing/updating requirements..."
pip install -r requirements.txt
