#!/bin/bash
set -e
echo "Testing all apps..."
libraries="mathy_python mathy_mkdocs mathy_pydoc_markdown website"
for library in $libraries
do
   echo "=== Testing: $library"
   (cd libraries/$library && sh tools/test.sh)
done


echo "Combining and reporting total mathy coverage:"
. ./libraries/website/.env/bin/activate
rm -rf .temp-cov
mkdir .temp-cov
cp ./libraries/mathy_python/.coverage ./.temp-cov/.coverage.mathy_python
cp ./libraries/website/.coverage ./.temp-cov/.coverage.website
cp ./libraries/website/.coveragerc ./.temp-cov/.coveragerc
cd .temp-cov
coverage combine
coverage report
rm -rf .temp-cov
cd -
