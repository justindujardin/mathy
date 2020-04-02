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


## get_model_meta
```python
get_model_meta(model_path:pathlib.Path)
```
Get model meta.json from a directory path and validate its contents.
path (unicode or Path): Path to model directory.

__Raises__

- `ValueError`: If **model_path** does not point to a valid folder.
- `ValueError`: If **model_path** does not have a `model.config.json` file.
- `ValueError`: If any required settings are missing from the meta file.

__Returns__

`(dict)`: The model's meta data.

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
load_model(name:Union[pathlib.Path, str], **overrides) -> mathy.api.Mathy
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

__Raises__

- `ValueError`: If the model path does not exist.

__Returns__

`(Language)`: `Language` class with loaded model.

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
) -> mathy.api.Mathy
```
Load a model from a data directory path.

__Arguments__

- __model_path (Path)__: The model folder to load from.

__Returns__

`(Mathy)`: a Mathy instance.

## package
```python
package(
    model_name: str,
    input_dir: Union[str, pathlib.Path],
    output_dir: Union[str, pathlib.Path],
    meta_path: Union[str, pathlib.Path] = None,
    force: bool = False,
) -> str
```
Generate a Python package from a Mathy model.

A new directory will be created in the specified output directory, and model data will be copied over.

__Arguments__

- __model_name__: the lower-case with underscores name for the model, e.g. "mathy_alpha_sm".
- __input_dir__: The folder containing a Mathy model to use as an input.
- __output_dir__: The folder to put the packaged model in.
- __force__: Delete any existing model in the output_dir rather than throw an error when this is true.

__Raises__

- `SystemExit`: If the input path does not exist.
- `SystemExit`: If the input path has no model.config.json file.
- `SystemExit`: If the model.config.json file is missing required keys.
- `SystemExit`: If output folder exists and `force` is False.

__Returns__

`(str)`: The subfolder of the output path that contains the pypi package source.

