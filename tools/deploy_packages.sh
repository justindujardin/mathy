#!/bin/bash
set -e

echo "Publishing all apps..."
libraries="mathy_python mathy_alpha_sm"
for library in $libraries
do
   echo "=== Deploying: $library"
   (cd libraries/$library && sh tools/deploy.sh)
done
