#!/bin/bash
set -e

sh tools/setup-ci.sh

# Make the virtualenv only if the folder doesn't exist
DIR=.env
if [ ! -d "${DIR}" ]; then
  echo "Installing root virtualenv (.env)"
  pip install virtualenv --upgrade
  python -m virtualenv .env -p python3.6
  echo "Installing/updating requirements..."
  .env/bin/pip install -e ./libraries/mathy_python
fi

. .env/bin/activate

