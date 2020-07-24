#!/bin/bash
set -e
echo "Testing all apps..."
libraries="mathy_python mathy_mkdocs website"
for library in $libraries
do
   echo "=== Testing: $library"
   (cd libraries/$library && sh tools/test.sh)
done


echo "Combining and reporting total mathy coverage:"
. .env/bin/activate
cp ./libraries/mathy_python/.coverage ./.coverage.mathy_python
cp ./libraries/website/.coverage ./.coverage.website
coverage combine
coverage report
coverage xml
coverage html
