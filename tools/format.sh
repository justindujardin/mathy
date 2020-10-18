#!/bin/sh -e
. .env/bin/activate

# Sort imports one per line, so autoflake can remove unused imports
isort libraries/mathy_python libraries/website --force-single-line-imports
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place libraries/mathy_python libraries/website --exclude=__init__.py
isort libraries/mathy_python libraries/website
black libraries/mathy_python libraries/website