from typing import Optional

import pytest

from simo_tools import generic_funcs as gf


class TestObj:
    A = "A"
    B = "B"
    C = "C"


@pytest.mark.parametrize(
    "ignore_attrs, expected_output",
    [
        pytest.param(None, "", id="ignore-none"),
        pytest.param(["C"], "", id="ignore-none"),
    ],
)
def test_build_repr(ignore_attrs: Optional[list[str]], expected_output: str):
    """
    generic_funcs.build_repr.
    """
    actual_result = gf.build_repr(TestObj(), ignore_attrs=ignore_attrs)
    breakpoint()
    assert actual_result == expected_output
