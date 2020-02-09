# Copyright (c) 2017  Niklas Rosenstein
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

from __future__ import print_function
from .document import Index
from .imp import import_object, dir_object
from argparse import ArgumentParser

import atexit
import os
import shutil
import signal
import subprocess
import sys
import yaml


parser = ArgumentParser()
parser.add_argument("subargs", nargs="...")


def default_config(config):
    args = parser.parse_args()
    config.setdefault("sort", "name")
    config.setdefault("headers", "markdown")
    config.setdefault("theme", "readthedocs")
    config.setdefault("loader", "mathy_pydoc.loader.PythonLoader")
    config.setdefault("preprocessor", "mathy_pydoc.preprocessor.Preprocessor")
    config.setdefault("additional_search_paths", [])
    return config


def makedirs(path):
    """Create the directory *path* if it does not already exist."""

    if not os.path.isdir(path):
        os.makedirs(path)


def log(*args, **kwargs):
    kwargs.setdefault("file", sys.stderr)
    print(*args, **kwargs)


def main():
    args = parser.parse_args()
    config = default_config({})
    # Parse options.
    modspecs = []
    it = iter(args.subargs)
    while True:
        try:
            value = next(it)
        except StopIteration:
            break
        if value == "-c":
            try:
                value = next(it)
            except StopIteration:
                parser.error("missing value to option -c")
            key, value = value.partition("=")[::2]
            if value.startswith("["):
                if not value.endswith("]"):
                    parser.error("invalid option value: {!r}".format(value))
                    value = value[1:-1].split(",")
            config[key] = value
        else:
            modspecs.append(value)
    args.subargs = modspecs

    loader = import_object(config["loader"])(config)
    preproc = import_object(config["preprocessor"])(config)

    # Build the index and document structure first, we load the actual
    # docstrings at a later point.
    log("Building index...")
    index = Index()

    def add_sections(doc, object_names, depth=1):
        if isinstance(object_names, list):
            [add_sections(doc, x, depth) for x in object_names]
        elif isinstance(object_names, dict):
            for key, subsections in object_names.items():
                add_sections(doc, key, depth)
                add_sections(doc, subsections, depth + 1)
        elif isinstance(object_names, str):
            # Check how many levels of recursion we should be going.
            expand_depth = len(object_names)
            object_names = object_names.rstrip("+")
            expand_depth -= len(object_names)

            def create_sections(name, level):
                if level > expand_depth:
                    return
                index.new_section(
                    doc,
                    name,
                    depth=depth + level,
                    header_type=config.get("headers", "html"),
                )
                sort_order = config.get("sort")
                if sort_order not in ("line", "name"):
                    sort_order = "line"
                need_docstrings = "docstring" in config.get("filter", ["docstring"])
                for sub in dir_object(name, sort_order, need_docstrings):
                    sub = name + "." + sub
                    sec = create_sections(sub, level + 1)

            create_sections(object_names, 0)
        else:
            raise RuntimeError(object_names)

    # Make sure that we can find modules from the current working directory,
    # and have them take precedence over installed modules.
    sys.path.insert(0, ".")

    # Generate a single document from the import names specified on the command-line.
    doc = index.new_document("main.md")
    add_sections(doc, args.subargs)

    # Load the docstrings and fill the sections.
    log("Started generating documentation...")
    for doc in index.documents.values():
        for section in filter(lambda s: s.identifier, doc.sections):
            loader.load_section(section)
            preproc.preprocess_section(section)

    for section in doc.sections:
        section.render(sys.stdout)
    return 0


if __name__ == "__main__":
    main()

