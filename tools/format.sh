#!/bin/sh -e
. .env/bin/activate

# Sort imports one per line, so autoflake can remove unused imports
isort mathy --force-single-line-imports
autoflake --remove-all-unused-imports --recursive --remove-unused-variables --in-place mathy --exclude=__init__.py
isort mathy
black mathy