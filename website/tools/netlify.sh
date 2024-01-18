#!/bin/bash
set -e
# Make the virtualenv only if the folder doesn't exist
pip install "mkdocs-material[imaging]"
echo "Building Netlify site..."
mkdocs build
