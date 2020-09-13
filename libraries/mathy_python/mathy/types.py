import sys

# Use typing_extensions for Python < 3.8
if sys.version_info < (3, 8):
    from typing_extensions import Final  # noqa
    from typing_extensions import Literal  # noqa
else:
    from typing import Final  # noqa
    from typing import Literal  # noqa
