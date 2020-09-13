from importlib.util import find_spec
from typing import Any, List, Union

import numpy as np

from mathy_envs import time_step


def has_modules(to_check: Union[List[str], str]) -> bool:
    if isinstance(to_check, str):
        return find_spec(to_check) is not None

    checks = [find_spec(a) is not None for a in to_check]
    return False not in checks


MISSING_LIBRARIES_ALERT = (
    "\n\nThe functionality you are trying to use requires optional "
    "packages that aren't installed:\n\n\t{}\n\nTry running:\n\n"
    "\tpip install mathy[{}]\n\n"
)
MODULE_JOIN = "\n\t"


def assert_fragile_installed():
    requires = ["fragile", "gym"]
    extra_name = "swarm"
    if not has_modules(requires):
        raise EnvironmentError(
            MISSING_LIBRARIES_ALERT.format(MODULE_JOIN.join(requires), extra_name)
        )

def is_terminal_transition(transition: time_step.TimeStep) -> bool:
    return bool(transition.step_type == time_step.StepType.LAST)


def pad_array(in_list: List[Any], max_length: int, value: Any = 0) -> List[Any]:
    """Pad a list to the given size with the given padding value.
    
    # Arguments:
    in_list (List[Any]): List of values to pad to the given length
    max_length (int): The desired length of the array
    value (Any): a value to insert in order to pad the array to max length

    # Returns
    (List[Any]): An array padded to `max_length` size
    """
    while len(in_list) < max_length:
        in_list.append(value)
    return in_list
