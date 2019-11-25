#!/bin/bash
set -e

# Make the virtualenv only if the folder doesn't exist
DIR=.env
if [ ! -d "${DIR}" ]; then
  virtualenv .env -p python3.6
fi

. .env/bin/activate
echo "Installing/updating requirements..."
pip install -e .[dev]

