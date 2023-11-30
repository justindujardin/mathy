"""Framework for FAI algorithms development."""

import warnings

warnings.filterwarnings(
    "ignore",
    message=(
        "Using or importing the ABCs from 'collections' instead of from 'collections.abc' "
        "is deprecated since Python 3.3,and in 3.9 it will stop working"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "the imp module is deprecated in favour of importlib; see the module's "
        "documentation for alternative uses"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "Using or importing the ABCs from 'collections' instead of from "
        "'collections.abc' is deprecated, and in 3.8 it will stop working"
    ),
)
warnings.filterwarnings(
    "ignore",
    message=(
        "The set_clim function was deprecated in Matplotlib 3.1 "
        "and will be removed in 3.3. Use ScalarMappable.set_clim "
        "instead."
    ),
)
warnings.filterwarnings("ignore", message="Gdk.Cursor.new is deprecated")

from .core import slogging  # noqa: E402
from .core.states import States  # noqa: E402
from .core.walkers import Walkers  # noqa: E402
from .version import __version__  # noqa: E402