# Based on spaCy's model loading utilities and CLI "package" command:
#  - https://github.com/explosion/spaCy/blob/master/spacy/util.py
#  - https://github.com/explosion/spaCy/blob/master/spacy/cli/package.py
"""Mathy implements a model loading and packaging scheme [based on the one from 
**spaCy**](https://spacy.io/){target=\_blank}. 

It can load models from **pypi packages** as well as **local folders** and **system links**.

!!! tip "spaCy is great for NLP tasks"

    Machine learning can often feel complex and hard to approach. spaCy is an open-source library
    for **Natural Language Processing** that is **ready for production**, **easy-to-use**, and **blazing fast**.

    With spaCy it's simple to get started working with text inputs:

    ```python
    import spacy

    nlp = spacy.load("en_core_web_sm")
    doc = nlp("justin and alex are both terrible at math")
    people = [t for t in doc if t.pos_ == "PROPN"]
    print(f"People found: {people}")
    ```

    Outputs:
    ```
    People found: [justin, alex]
    ```

    By looking at the **Parts of Speech** (referenced as `t.pos_`) we can find the people
    referenced in input.

    spaCy does much more than this, with examples that run inside the website. 
    Check it out: **[https://spacy.io](https://spacy.io){target=\_blank}**

"""
import importlib
import shutil
from pathlib import Path
from typing import Any, Optional, Union

import srsly
from wasabi import get_raw_input, msg

from . import about
from .api import Mathy

_data_path = Path(__file__).parent / "data"


REQUIRED_META_KEYS = ["units", "embedding_units", "lstm_units", "version"]
REQUIRED_MODEL_FILES = ["model.h5", "model.optimizer", "model.config.json"]


def load_model(name: Union[Path, str], **overrides) -> Mathy:
    """Load a model from a shortcut link, package or data path.

    # Arguments
    name (str): Package name, shortcut link or model path.
    overrides (kwargs): Specific overrides, like how many MCTS sims to use

    # Raises
    ValueError: If *name* is not recognized loadable package or model folder.

    # Returns
    (Mathy): Mathy class with the loaded model.
    """
    if isinstance(name, str):
        if is_package(name):  # installed as package
            return load_model_from_package(name, **overrides)
        if Path(name).exists():  # path to model data directory
            return load_model_from_path(Path(name), **overrides)
    elif hasattr(name, "exists"):  # Path or Path-like to model data
        return load_model_from_path(name, **overrides)
    raise ValueError(f"Unrecognized model input: {name}")


def load_model_from_package(name, **overrides):
    """Load a model from an installed package."""
    cls = importlib.import_module(name)
    return cls.load(**overrides)


def load_model_from_path(model_path: Path, meta: dict = None, **overrides) -> Mathy:
    """Load a model from a data directory path.
    
    # Arguments
    model_path (Path): The model folder to load from.

    # Returns
    (Mathy): a Mathy instance.
    """
    if not meta:
        meta = get_model_meta(model_path)
    mt = Mathy(model_path=str(model_path), **overrides)
    return mt


def load_model_from_init_py(init_file: Union[Path, str], **overrides):
    """Helper function to use in the `load()` method of a model package's
    __init__.py.

    # Arguments
    init_file (unicode): Path to model's __init__.py, i.e. `__file__`.
    **overrides: Specific overrides, like pipeline components to disable.

    # Raises
    ValueError: If the model path does not exist.
    
    # Returns
    (Language): `Language` class with loaded model.
    """
    model_path = Path(init_file).parent
    if not model_path.exists():
        raise ValueError(f"model path does not exist: {model_path}")
    meta = get_model_meta(model_path)
    data_path = model_path
    return load_model_from_path(data_path, meta, **overrides)


def get_model_meta(model_path: Path):
    """Get model meta.json from a directory path and validate its contents.
    path (unicode or Path): Path to model directory.

    # Raises
    ValueError: If **model_path** does not point to a valid folder.
    ValueError: If **model_path** does not have a `model.config.json` file.
    ValueError: If any required settings are missing from the meta file.

    # Returns
    (dict): The model's meta data.
    """
    model_path = Path(model_path)
    if not model_path.exists():
        raise ValueError(f"cannot get meta from invalid model path: {model_path}")
    meta_path = model_path / "model.config.json"
    if not meta_path.is_file():
        raise ValueError(f"invalid meta file: {meta_path}")
    meta = srsly.read_json(meta_path)
    for setting in REQUIRED_META_KEYS:
        if setting not in meta or not meta[setting]:
            raise ValueError(f"meta file missing required setting: {setting}")
    return meta


def is_package(name: str) -> bool:
    """Check if string maps to a package installed via pip.

    # Arguments
    name (str): Name of package.

    # Returns
    (bool): True if installed package, False if not.
    """
    import pkg_resources

    name = name.lower()  # compare package name against lowercase name
    packages = pkg_resources.working_set.by_key.keys()  # type:ignore
    for package in packages:
        if package.lower().replace("-", "_") == name:
            return True
    return False


def package(
    model_name: str,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    meta_path: Union[str, Path] = None,
    force: bool = False,
) -> str:
    """Generate a Python package from a Mathy model.
    
    A new directory will be created in the specified output directory, and model data will be copied over.

    # Arguments
    model_name: the lower-case with underscores name for the model, e.g. "mathy_alpha_sm".
    input_dir: The folder containing a Mathy model to use as an input.
    output_dir: The folder to put the packaged model in.
    force: Delete any existing model in the output_dir rather than throw an error when this is true.

    # Raises
    SystemExit: If the input path does not exist.
    SystemExit: If the input path has no model.config.json file.
    SystemExit: If the model.config.json file is missing required keys.
    SystemExit: If output folder exists and `force` is False.

    # Returns
    (str): The subfolder of the output path that contains the pypi package source.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if meta_path is not None:
        meta_path = Path(meta_path)
    if not input_path or not input_path.exists():
        msg.fail("Can't locate model data", input_path, exits=1)
    if meta_path and not meta_path.exists():
        msg.fail("Can't find model model.config.json", meta_path, exits=1)
    meta_path = meta_path or input_path / "model.config.json"
    if meta_path.is_file():
        meta = srsly.read_json(meta_path)
        msg.good("Loaded model.config.json from file", meta_path)
    meta["mathy_version"] = f">={about.__version__},<1.0.0"
    meta["name"] = model_name
    for key in REQUIRED_META_KEYS:
        if key not in meta or meta[key] == "":
            msg.fail(
                "No '{}' setting found in model.config.json".format(key),
                "This setting is required to build your package.",
                exits=1,
            )
    main_path = output_path / model_name
    package_path = main_path
    if package_path.exists():
        if force:
            shutil.rmtree(str(package_path))
        else:
            msg.fail(
                title="Package directory already exists",
                text="Please delete the directory and try again, or use the "
                "`--force` flag to overwrite existing directories.",
                exits=1,
            )
    Path.mkdir(package_path, parents=True, exist_ok=True)
    for f in REQUIRED_MODEL_FILES:
        file_name: Path = input_path / f
        if not file_name.exists():
            msg.fail(
                f"Input path '{input_path}' is missing a required file: '{f}'",
                "This file is required to build your package.",
                exits=1,
            )
        shutil.copyfile(file_name, main_path / f)
    create_file(output_path / "model.config.json", srsly.json_dumps(meta, indent=2))
    create_file(output_path / "setup.py", TEMPLATE_SETUP)
    create_file(package_path / "__init__.py", TEMPLATE_INIT)
    msg.good("Successfully created package '{}'".format(package_path), main_path)
    msg.text("To build the package, run `python setup.py sdist` in this directory.")
    return str(package_path)


def create_file(file_path: Path, contents: str):
    file_path.touch()
    file_path.open("w", encoding="utf-8").write(contents)


TEMPLATE_SETUP = """
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
    parent_package = meta.get("parent_package", "mathy")
    requirements = [parent_package + meta["mathy_version"]]
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
    setup(
        name=model_name,
        description=meta["description"],
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
    setup_package()""".strip()


TEMPLATE_INIT = """
# coding: utf8
from __future__ import unicode_literals
from pathlib import Path
from mathy.models import load_model_from_init_py, get_model_meta
__version__ = get_model_meta(Path(__file__).parent)['version']
def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)
""".strip()
