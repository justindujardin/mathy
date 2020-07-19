#!/bin/bash
set -e

. ../../.env/bin/activate
git config --global user.email "justin@dujardinconsulting.com"
git config --global user.name "justindujardin"
echo "Build and publish to pypi..."
rm -rf build dist
echo "--- Install requirements"
pip install twine wheel
echo "--- Buid dists"
python setup.py sdist bdist_wheel
echo "--- Upload to PyPi"
# NOTE: ignore errors on upload because our CI is dumb and tries to upload
#       even if the version has already been uploaded. This isn't great, but
#       works for now. Ideally the CI would not call this script unless the
#       semver changed.
set +e
twine upload -u ${PYPI_USERNAME} -p ${PYPI_PASSWORD} dist/*.whl
rm -rf build dist
