"""Module that implements classes that use ray."""
import sys

try:
    import ray
except (ImportError, ModuleNotFoundError) as e:
    if sys.version_info <= (3, 7):
        raise e
    else:

        class ray:
            """Dummy to avoid import errors before ray is released for Python 3.8."""

            def remote(self, *args, **kwargs):
                """Do nothing."""
                pass
