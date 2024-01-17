#!/bin/bash
sh ./tools/setup.sh
echo "Building Netlify site..."
mkdocs build
