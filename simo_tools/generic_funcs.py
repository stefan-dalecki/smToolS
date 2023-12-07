"""
Functions that are not specific to any other module or module theme.
"""

from typing import Any, Optional


def build_repr(obj: Any, *, ignore_attrs: Optional[list[str]] = None) -> str:
    """
    Creates string of class name with all attributes.
    """
    ignore_attrs = ignore_attrs or []
    attrs = []
    for attr, val in obj.__dict__.items():
        if not attr.startswith("_") and attr not in ignore_attrs:
            attrs += [f"{attr}={val}"]
    return f"{obj.__class__.__name__}({', '.join(attrs)})"
