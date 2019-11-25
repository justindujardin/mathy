#!/bin/bash
set -e
echo "Testing all apps..."
libraries="mathy_python website"
for library in $libraries
do
   echo "=== Testing: $library"
   (cd libraries/$library && sh tools/test.sh)
done
