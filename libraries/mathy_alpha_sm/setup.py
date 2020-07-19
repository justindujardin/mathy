#!/usr/bin/env python
# coding: utf8
from typing import List
import io
import json
from os import path, walk
from shutil import copy
from setuptools import setup


def load_meta(fp):
    with io.open(fp, encoding="utf8") as f:
        return json.load(f)


def list_files(data_dir: str, root: str) -> List[str]:
    output = []
    package_dir = path.join(root, data_dir)
    for folder, _, filenames in walk(package_dir):
        if "__pycache__" in folder:
            continue
        for filename in filenames:
            if not filename.startswith("."):
                output.append(path.join(folder, filename))
    rel_output = [path.relpath(p, package_dir) for p in output]
    return rel_output


def list_requirements(meta):
    parent_package = meta.get("parent_package", "mathy[tf]")
    requirements = [parent_package + meta["mathy_version"]]
    requirements += ["tf_siren>=0.0.3"]
    if "setup_requires" in meta:
        requirements += meta["setup_requires"]
    if "requirements" in meta:
        requirements += meta["requirements"]
    return requirements


def setup_package():
    root = path.abspath(path.dirname(__file__))
    meta_path = path.join(root, "model.config.json")
    meta = load_meta(meta_path)
    model_name = meta["name"]
    model_dir = path.join(model_name)
    # Copy in mathy readme
    with open(path.join(root, "../../README.md"), "r") as fh:
        long_description = fh.read()
    setup(
        name=model_name,
        description=meta["description"],
        long_description=long_description,
        long_description_content_type="text/markdown",
        author=meta["author"],
        author_email=meta["email"],
        url=meta["url"],
        version=meta["version"],
        license=meta["license"],
        packages=[model_name],
        package_data={model_name: list_files(model_dir, root)},
        install_requires=list_requirements(meta),
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
