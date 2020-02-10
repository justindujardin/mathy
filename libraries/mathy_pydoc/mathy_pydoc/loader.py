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
"""
This module provides implementations to load documentation information from
an identifier as it is specified in the `pydocmd.yml:generate` configuration
key. A loader basically takes care of loading the documentation content for
that name, but is not supposed to apply preprocessing.
"""

from __future__ import print_function
from .imp import import_object_with_scope
import inspect
import types
from typing import Callable, Optional, List, Any
import re

function_types = (
    types.FunctionType,
    types.LambdaType,
    types.MethodType,
    types.BuiltinFunctionType,
    types.BuiltinMethodType,
)
if hasattr(types, "UnboundMethodType"):
    function_types += (types.UnboundMethodType,)

# Union[MathyEnvState, NoneType] -> Optional[MathyEnvState]
optional_match = r"(.*)Union\[(.*),\sNoneType\](.*)"
optional_replace = r"\1Optional[\2]\3"

# _ForwardRef('MathyEnvState') -> MathyEnvState
fwd_ref_match = r"(.*)\_ForwardRef\(\'(.*)\'\)(.*)"
fwd_ref_replace = r"\1\2\3"


def cleanup_type(type_string: str) -> str:
    # Optional[T] gets expanded to Union[T, NoneType], so change it back
    while re.search(optional_match, type_string) is not None:
        type_string = re.sub(optional_match, optional_replace, type_string)

    # _ForwardRef('MathyEnvState') -> MathyEnvState
    while re.search(fwd_ref_match, type_string) is not None:
        type_string = re.sub(fwd_ref_match, fwd_ref_replace, type_string)

    return type_string


def trim(docstring):
    if not docstring:
        return ""
    lines = [x.rstrip() for x in docstring.split("\n")]
    lines[0] = lines[0].lstrip()

    indent = None
    for i, line in enumerate(lines):
        if i == 0 or not line:
            continue
        new_line = line.lstrip()
        delta = len(line) - len(new_line)
        if indent is None:
            indent = delta
        elif delta > indent:
            new_line = " " * (delta - indent) + new_line
        lines[i] = new_line

    return "\n".join(lines)


class PythonLoader(object):
    """
  Expects absolute identifiers to import with #import_object_with_scope().
  """

    def __init__(self, config):
        self.config = config

    def load_section(self, section):
        """
    Loads the contents of a #Section. The `section.identifier` is the name
    of the object that we need to load.

    # Arguments
      section (Section): The section to load. Fill the `section.title` and
        `section.content` values. Optionally, `section.loader_context` can
        be filled with custom arbitrary data to reference at a later point.
    """

        assert section.identifier is not None
        obj, scope = import_object_with_scope(section.identifier)

        if "." in section.identifier:
            default_title = section.identifier.rsplit(".", 1)[1]
        else:
            default_title = section.identifier

        section.title = getattr(obj, "__name__", default_title)
        section.content = trim(get_docstring(obj))
        section.loader_context = {"obj": obj, "scope": scope}

        # Add the function signature in a code-block.
        if callable(obj):
            sig = get_function_signature(obj, scope if inspect.isclass(scope) else None)
            section.content = "```python\n{}\n```\n".format(sig) + section.content


def get_docstring(function):
    if hasattr(function, "__name__") or isinstance(function, property):
        return function.__doc__ or ""
    elif hasattr(function, "__call__"):
        return function.__call__.__doc__ or ""
    else:
        return function.__doc__ or ""


class CallableArg:
    name: str
    type_hint: Optional[str]
    default: Optional[str]

    def __init__(self, name: str, type_hint: Optional[str], default: Optional[str]):
        self.name = name
        self.type_hint = type_hint
        self.default = default


class CallablePlaceholder:
    simple: str
    name: str
    args: List[CallableArg]
    return_type: Optional[str]

    def __init__(
        self,
        simple: str,
        name: str,
        args: List[CallableArg],
        return_type: Optional[Any] = None,
    ):
        self.simple = simple
        self.name = name
        self.args = args
        self.return_type = return_type


def get_callable_placeholder(
    function: Callable, owner_class=None, show_module=False
) -> CallablePlaceholder:
    isclass = inspect.isclass(function)

    # Get base name.
    name_parts = []
    if show_module:
        name_parts.append(function.__module__)
    if owner_class:
        name_parts.append(owner_class.__name__)
    if hasattr(function, "__name__"):
        name_parts.append(function.__name__)
    else:
        name_parts.append(type(function).__name__)
        name_parts.append("__call__")
        function = function.__call__
    if isclass:
        function = getattr(function, "__init__", None)

    name = ".".join(name_parts)
    sig = inspect.signature(function)

    params = []
    for p in sig.parameters.values():
        annotation = None
        default_value = None
        if p.annotation is not inspect._empty:  # type: ignore
            annotation = inspect.formatannotation(p.annotation)
        if p.default is not inspect._empty:  # type: ignore
            if isinstance(p.default, str):
                default_value = repr(p.default)
            else:
                default_value = str(p.default)
        if annotation is not None:
            annotation = cleanup_type(annotation)
        params.append(CallableArg(p.name, annotation, default_value))

    return_annotation = None
    if sig.return_annotation is not inspect._empty:  # type: ignore
        return_annotation = inspect.formatannotation(
            sig.return_annotation, base_module="mathy"
        )
    if return_annotation is not None:
        return_annotation = cleanup_type(return_annotation)
    return CallablePlaceholder(
        simple=str(sig), name=name, args=params, return_type=return_annotation
    )


def get_function_signature(
    function: Callable,
    owner_class: Optional[Any] = None,
    show_module: bool = False,
    indent: int = 4,
    max_width: int = 82,
) -> str:
    isclass = inspect.isclass(function)

    placeholder: CallablePlaceholder = get_callable_placeholder(
        function=function, owner_class=owner_class, show_module=show_module
    )

    out_str = placeholder.name + placeholder.simple
    if len(out_str) < max_width:
        return out_str
    out_str = f"{placeholder.name}(\n"
    arg: CallableArg
    indent = " " * indent
    for arg in placeholder.args:
        arg_str = f"{indent}{arg.name}"
        if arg.type_hint is not None:
            arg_str += f": {arg.type_hint}"
        if arg.default is not None:
            arg_str += f" = {arg.default}"
        arg_str += ",\n"
        out_str += arg_str
    out_str += f")"
    if placeholder.return_type is not None:
        out_str += f" -> {placeholder.return_type}"

    return out_str
