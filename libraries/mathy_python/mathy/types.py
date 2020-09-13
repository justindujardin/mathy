import sys
from typing import List, NamedTuple, Tuple
from enum import Enum
from pydantic import BaseModel, Field

# Use typing_extensions for Python < 3.8
if sys.version_info < (3, 8):
    from typing_extensions import Final, Literal
else:
    from typing_extensions import Final, Literal  # noqa

