#!/bin/bash
set -e

echo "Activating virtualenv... (if this fails you may need to run setup.sh first)"

echo "Running tests..."
.env/bin/python -m pytest --cov=mathy

(cd website && sh tools/test.sh)

echo "Combining and reporting total mathy coverage:"
. .env/bin/activate
cp ./.coverage ./.coverage.mathy_python
cp ./website/.coverage ./.coverage.website
coverage combine
coverage report
coverage xml
coverage html
