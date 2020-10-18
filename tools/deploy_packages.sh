#!/bin/bash
set -e

echo "Publishing all apps..."
libraries="mathy_python"
for library in $libraries
do
   echo "=== Deploying: $library"
   (cd libraries/$library && sh tools/deploy.sh)
done
