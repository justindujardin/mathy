#!/bin/bash
set -e
echo "Cleaning all setup/build files for apps..."
echo "You will have to run the root 'sh tools/setup.sh' again after this."
libraries="mathy_python mathy_mkdocs website"
for library in $libraries
do
   echo "=== Cleaning: $library"
   (cd libraries/$library && sh tools/clean.sh)
done
