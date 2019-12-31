# mathy.models
Mathy implements a model loading and packaging scheme [based on the one from
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


## get_data_path
```python
get_data_path(require_exists=True) -> Union[pathlib.Path, NoneType]
```
Get path to Mathy data directory.

__Arguments__

- __require_exists (bool)__: Only return path if it exists, otherwise None.

__Returns__

`(Optional[Path])`: Data path or None.

## get_model_meta
```python
get_model_meta(model_path:pathlib.Path)
```
Get model meta.json from a directory path and validate its contents.
path (unicode or Path): Path to model directory.

__Raises__

- `ValueError`: If **model_path** does not point to a valid folder
- `ValueError`: If **model_path** does not have a `model.config.json` file
- `ValueError`: If any required settings are missing from the meta file

__Returns__

`(dict)`: The model's meta data.

## get_package_path
```python
get_package_path(name:str) -> pathlib.Path
```
Get the path to an installed package.

__Arguments__

- __name (unicode)__: Package name.

__Returns__

`(Path)`: Path to installed package.

## is_package
```python
is_package(name:str) -> bool
```
Check if string maps to a package installed via pip.

__Arguments__

- __name (str)__: Name of package.

__Returns__

`(bool)`: True if installed package, False if not.

## load_model
```python
load_model(name:str, **overrides) -> mathy.mathy.Mathy
```
Load a model from a shortcut link, package or data path.

__Arguments__

- __name (str)__: Package name, shortcut link or model path.
- __overrides (kwargs)__: Specific overrides, like how many MCTS sims to use

__Raises__

- `ValueError`: If *name* is not recognized loadable package or model folder.

__Returns__

`(Mathy)`: Mathy class with the loaded model.

## load_model_from_init_py
```python
load_model_from_init_py(init_file:Union[pathlib.Path, str], **overrides)
```
Helper function to use in the `load()` method of a model package's
__init__.py.

__Arguments__

- __init_file (unicode)__: Path to model's __init__.py, i.e. `__file__`.
- __**overrides__: Specific overrides, like pipeline components to disable.

__Returns__

`(Language)`: `Language` class with loaded model.

## load_model_from_link
```python
load_model_from_link(name:str, **overrides) -> mathy.mathy.Mathy
```
Load a model from a shortcut link, or directory in Mathy data path.

__Arguments__

- __name (str)__: Package name, shortcut link or model path.
- __overrides (kwargs)__: Specific overrides, like how many MCTS sims to use

## load_model_from_package
```python
load_model_from_package(name, **overrides)
```
Load a model from an installed package.
## load_model_from_path
```python
load_model_from_path(
    model_path: pathlib.Path,
    meta: dict = None,
    overrides,
) -> mathy.mathy.Mathy
```
Load a model from a data directory path.
## package
```python
package(
    model_name: str,
    input_dir: Union[str, pathlib.Path],
    output_dir: Union[str, pathlib.Path],
    meta_path: Union[str, pathlib.Path] = None,
    create_meta: bool = False,
    force: bool = False,
) -> str
```

Generate Python package for model data, including meta and required
installation files. A new directory will be created in the specified
output directory, and model data will be copied over. If --create-meta is
set and a model.config.json already exists in the output directory, the existing
values will be used as the defaults in the command-line prompt.

## set_data_path
```python
set_data_path(path)
```
Set path to Mathy data directory.

__Arguments:__

path (unicode or Path): Path to new data directory.

