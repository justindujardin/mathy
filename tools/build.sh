#!/bin/bash
echo "Building all apps..."
set -e
libraries="mathy_python website"
for library in $libraries
do
   echo "=== Building: $library"
   (cd libraries/$library && sh tools/build.sh)
done
