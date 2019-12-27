#!/bin/bash
set -e


echo "Setting up all apps..."
libraries="mathy_python mathy_mkdocs mathy_pydoc_markdown website"
for library in $libraries
do
   echo "=== Setting up: $library"
   (cd libraries/$library && sh tools/setup.sh)
done


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

