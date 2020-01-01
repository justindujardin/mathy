#!/bin/bash
echo "Installing semantic-release requirements"
npm install 
echo "Updating build version"
npx ts-node tools/set-build-version.ts
echo "Running semantic-release"
npx semantic-release
