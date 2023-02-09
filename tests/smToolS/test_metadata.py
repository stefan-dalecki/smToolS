import datetime
import os
import sys
from typing import AnyStr, List, Optional, Union

import pandas as pd
import pytest

script_path = os.path.realpath(__file__)
tool_path = os.path.realpath(os.path.join(script_path, "..", "..", ".."))
sys.path.insert(0, tool_path)

from smToolS import metadata
from smToolS.sm_helpers import constants as cons


class TestMetadataFunctions:
    df: pd.DataFrame

    @pytest.fixture(autouse=True)
    def setup_and_tear_down(self):
        self._reorder_df = pd.DataFrame(data={"abc": [1, 2, 3], "def": [3, 4, 5], "ghi": [6, 7, 8]})

    @pytest.mark.parametrize(
        "date_str, failure_str, expected_return",
        [
            ("abc\\123\\2023_02_05\\abc\\123", "this will not fail", datetime.date(2023, 2, 5)),
            ("abc\123", "this_will_fail", "this_will_fail"),
        ],
        ids=["successfully locate date", "return specified failure message"],
    )
    def test_find_date(self, date_str: str, failure_str, expected_return):
        actual_return = metadata.find_date(date_str, failure=failure_str)
        assert actual_return[cons.DATE] == expected_return

    @pytest.mark.parametrize(
        "full_str, sep, value_name, value_search_names, failure, expected_str",
        [
            (
                r"2021\2021_11_02\gas1\67pM-GRP1_ND06_01",
                os.sep,
                cons.GASKET,
                [cons.GASKET_ABV],
                "this will not fail",
                "1",
            ),
            (
                r"2021\2021_11_02\gas1\67pM-GRP1_ND06_01",
                "_",
                cons.GASKET,
                [cons.GASKET_ABV],
                "this will fail",
                "this will fail",
            ),
        ],
    )
    def test_find_identifiers(
        self,
        full_str: AnyStr,
        sep: AnyStr,
        value_name: AnyStr,
        value_search_names: List[AnyStr],
        failure: AnyStr,
        expected_str: AnyStr,
    ):
        actual_return = metadata.find_identifiers(
            full_str, sep, value_name, value_search_names, failure=failure
        )
        expected_return = {value_name: expected_str}
        assert actual_return == expected_return

    @pytest.mark.parametrize(
        "column_name, location, expected_output",
        [("abc", 2, 2), ("nonexistent_column", 2, KeyError)],
        ids=["successful reorder", "reorder raises KeyError"],
    )
    def test_reorder(
        self, column_name: AnyStr, location: int, expected_output: Union[AnyStr, Exception]
    ):
        if isinstance(expected_output, type) and issubclass(expected_output, Exception):
            with pytest.raises(expected_output):
                _ = metadata.reorder(self._reorder_df, column_name, location)
        else:
            reordered_df = metadata.reorder(self._reorder_df, column_name, location)
            actual_column_location = reordered_df.columns.get_loc(column_name)
            assert expected_output == actual_column_location

    @pytest.mark.parametrize(
        "item, options, expected_output",
        [("abc", ["abc", "def"], None), ("abc", ["123", "456", "789"], KeyError)],
    )
    def test__attribute_validation(
        self, item: AnyStr, options: List[AnyStr], expected_output: Optional[Exception]
    ):
        if isinstance(expected_output, type) and issubclass(expected_output, Exception):
            with pytest.raises(expected_output):
                _ = metadata._attribute_validation(item, options)
        else:
            _ = metadata._attribute_validation(item, options)
