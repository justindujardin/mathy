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
from .mathy import Mathy

_data_path = Path(__file__).parent / "data"


REQUIRED_META_KEYS = ["units", "embedding_units", "lstm_units", "version"]


def get_data_path(require_exists=True) -> Optional[Path]:
    """Get path to Mathy data directory.

    # Arguments
    require_exists (bool): Only return path if it exists, otherwise None.

    # Returns
    (Optional[Path]): Data path or None.
    """
    if not require_exists:
        return _data_path
    else:
        return _data_path if _data_path.exists() else None


def set_data_path(path):
    """Set path to Mathy data directory.

    # Arguments:
    path (unicode or Path): Path to new data directory.
    """
    global _data_path
    _data_path = ensure_path(path)


def load_model(name: str, **overrides) -> Mathy:
    """Load a model from a shortcut link, package or data path.

    # Arguments
    name (str): Package name, shortcut link or model path.
    overrides (kwargs): Specific overrides, like how many MCTS sims to use

    # Raises
    ValueError: If *name* is not recognized loadable package or model folder.

    # Returns
    (Mathy): Mathy class with the loaded model.
    """
    if isinstance(name, str):  # in data dir / shortcut
        if is_package(name):  # installed as package
            return load_model_from_package(name, **overrides)
        if Path(name).exists():  # path to model data directory
            return load_model_from_path(Path(name), **overrides)
    elif hasattr(name, "exists"):  # Path or Path-like to model data
        return load_model_from_path(name, **overrides)
    raise ValueError(f"Unrecognized model input: {name}")


def load_model_from_link(name: str, **overrides) -> Mathy:
    """Load a model from a shortcut link, or directory in Mathy data path.
    
    # Arguments
    name (str): Package name, shortcut link or model path.
    overrides (kwargs): Specific overrides, like how many MCTS sims to use
    """
    data_path = get_data_path()
    assert data_path is not None
    import_path = data_path / name / "__init__.py"
    try:
        import importlib.util

        spec = importlib.util.spec_from_file_location(name, str(import_path))
        cls: Any = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)  # type: ignore
    except AttributeError:
        raise ValueError(f"Invalid module '{name}' at: {str(import_path)}")
    return cls.load(**overrides)


def load_model_from_package(name, **overrides):
    """Load a model from an installed package."""
    cls = importlib.import_module(name)
    return cls.load(**overrides)


def load_model_from_path(model_path: Path, meta: dict = None, **overrides) -> Mathy:
    """Load a model from a data directory path."""
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
    
    # Returns
    (Language): `Language` class with loaded model.
    """
    model_path = Path(init_file).parent
    meta = get_model_meta(model_path)
    data_path = model_path / f"{meta['name']}-{meta['version']}"
    if not model_path.exists():
        raise ValueError(f"model path does not exist: {model_path}")
    return load_model_from_path(data_path, meta, **overrides)


def get_model_meta(model_path: Path):
    """Get model meta.json from a directory path and validate its contents.
    path (unicode or Path): Path to model directory.

    # Raises
    ValueError: If **model_path** does not point to a valid folder
    ValueError: If **model_path** does not have a `model.config.json` file
    ValueError: If any required settings are missing from the meta file

    # Returns
    (dict): The model's meta data.
    """
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


def get_package_path(name: str) -> Path:
    """Get the path to an installed package.

    # Arguments
    name (unicode): Package name.

    # Returns
    (Path): Path to installed package.
    """
    name = name.lower()  # use lowercase version to be safe
    # Here we're importing the module just to find it. This is worryingly
    # indirect, but it's otherwise very difficult to find the package.
    pkg = importlib.import_module(name)
    return Path(pkg.__file__).parent


def package(
    model_name: str,
    input_dir: Union[str, Path],
    output_dir: Union[str, Path],
    meta_path: Union[str, Path] = None,
    create_meta: bool = False,
    force: bool = False,
) -> str:
    """
    Generate Python package for model data, including meta and required
    installation files. A new directory will be created in the specified
    output directory, and model data will be copied over. If --create-meta is
    set and a model.config.json already exists in the output directory, the existing
    values will be used as the defaults in the command-line prompt.
    """
    input_path = Path(input_dir)
    output_path = Path(output_dir)
    if meta_path is not None:
        meta_path = Path(meta_path)
    if not input_path or not input_path.exists():
        msg.fail("Can't locate model data", input_path, exits=1)
    if not output_path or not output_path.exists():
        msg.fail("Output directory not found", output_path, exits=1)
    if meta_path and not meta_path.exists():
        msg.fail("Can't find model model.config.json", meta_path, exits=1)

    meta_path = meta_path or input_path / "model.config.json"
    if meta_path.is_file():
        meta = srsly.read_json(meta_path)
        msg.good("Loaded model.config.json from file", meta_path)
    meta["mathy_version"] = f">={about.__version__},<1.0.0"
    for key in REQUIRED_META_KEYS:
        if key not in meta or meta[key] == "":
            msg.fail(
                "No '{}' setting found in model.config.json".format(key),
                "This setting is required to build your package.",
                exits=1,
            )
    main_path = output_path
    package_path = main_path / model_name
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
    Path.mkdir(main_path, parents=True, exist_ok=True)
    shutil.copytree(str(input_path), str(package_path))
    meta["name"] = model_name
    create_file(main_path / "model.config.json", srsly.json_dumps(meta, indent=2))
    create_file(main_path / "setup.py", TEMPLATE_SETUP)
    create_file(main_path / "MANIFEST.in", TEMPLATE_MANIFEST)
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
from __future__ import unicode_literals
import io
import json
from os import path, walk
from shutil import copy
from setuptools import setup


def load_meta(fp):
    with io.open(fp, encoding="utf8") as f:
        return json.load(f)


def list_files(data_dir):
    output = []
    for root, _, filenames in walk(data_dir):
        for filename in filenames:
            if not filename.startswith("."):
                output.append(path.join(root, filename))
    output = [path.relpath(p, path.dirname(data_dir)) for p in output]
    output.append("model.config.json")
    return output


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
    model_dir = path.join(model_name, model_name + "-" + str(meta["version"]))
    copy(meta_path, path.join(model_name))
    copy(meta_path, model_dir)
    setup(
        name=model_name,
        description=meta["description"],
        author=meta["author"],
        author_email=meta["email"],
        url=meta["url"],
        version=meta["version"],
        license=meta["license"],
        packages=[model_name],
        package_data={model_name: list_files(model_dir)},
        install_requires=list_requirements(meta),
        zip_safe=False,
    )


if __name__ == "__main__":
    setup_package()
""".strip()


TEMPLATE_MANIFEST = """
include model.config.json
""".strip()


TEMPLATE_INIT = """
# coding: utf8
from __future__ import unicode_literals
from pathlib import Path
from mathy.util import load_model_from_init_py, get_model_meta
__version__ = get_model_meta(Path(__file__).parent)['version']
def load(**overrides):
    return load_model_from_init_py(__file__, **overrides)
""".strip()
