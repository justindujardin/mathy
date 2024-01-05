#!/bin/bash
set -e
. ../.env/bin/activate
echo "Build python package..."
mkdocs build
