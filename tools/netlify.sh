#!/usr/bin/env bash
pip install virtualenv

cd libraries/website

sh tools/setup.sh

sh tools/build.sh
