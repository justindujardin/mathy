#!/bin/bash
set -e

. ../.env/bin/activate

mkdocs serve
