"""General utility functions."""

import importlib.util
import logging

logger = logging.getLogger(__name__)


def is_module_installed(module_name: str, *, throw_error: bool = False) -> bool:
    """Check if the module is installed or not.

    Examples:
        >>> is_module_installed(module_name="yaml", throw_error=False)
        True
        >>> is_module_installed(module_name="numpy", throw_error=False)
        False
        >>> is_module_installed(module_name="numpy", throw_error=True)
        Traceback (most recent call last):
        ImportError: Module numpy is not installed.

    Args:
        module_name: Name of the module to be checked.
        throw_error: If True, raises ImportError if module is not installed.

    Returns:
        Returns True if module is installed, False otherwise.

    Raises:
        ImportError: If throw_error is True and module is not installed.
    """

    is_installed = importlib.util.find_spec(module_name) is not None
    if not is_installed and throw_error:
        raise ImportError(f"Module {module_name} is not installed.")
    return is_installed
