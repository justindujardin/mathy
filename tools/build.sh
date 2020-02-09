#!/bin/bash
echo "Building all apps..."
set -e
libraries="mathy_python mathy_alpha_sm mathy_mkdocs mathy_pydoc website"
for library in $libraries
do
   echo "=== Building: $library"
   (cd libraries/$library && sh tools/build.sh)
done
