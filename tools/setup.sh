#!/bin/bash
set -e

sh tools/ci-setup.sh

# Make the virtualenv only if the folder doesn't exist
DIR=.env
if [ ! -d "${DIR}" ]; then
  echo "Installing root virtualenv (.env)"
  pip install virtualenv --upgrade
  python -m virtualenv .env -p python3.6
fi
echo "Installing/updating requirements..."
.env/bin/pip install -e ./libraries/mathy_python[all]

. .env/bin/activate

