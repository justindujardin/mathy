import os
import tempfile
from pathlib import Path
from shutil import copyfile

import ujson

from ...environment_state import INPUT_EXAMPLES_FILE_NAME


def memory_files(from_folder: str):
    """Get a list of fully-qualified filenames for all the JSONL memory
    groups that exist in a path.
    
    Returns: string list of files"""
    groups = []
    for filename in os.listdir(from_folder):
        full_file = Path(os.path.join(from_folder, filename))
        if full_file.is_file() and full_file.suffix == "jsonl":
            groups.append(str(full_file))


class MathMemory:
    """Aggregate experience across a number of problem types, and present it in 
    varied ways to the agent for training. """
    pass
