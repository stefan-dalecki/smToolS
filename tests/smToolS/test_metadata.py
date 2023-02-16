import datetime
import os
from typing import AnyStr, List, Optional, Tuple, Union
from unittest.mock import PropertyMock, patch

import pandas as pd
import pytest

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
            (
                "abc\\123\\2023_02_05\\abc\\123",
                "this will not fail",
                datetime.date(2023, 2, 5),
            ),
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
        ids=["find sub-str in str", "ensure failure argument usage"],
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
        self,
        column_name: AnyStr,
        location: int,
        expected_output: Union[AnyStr, Exception],
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
        ids=["valid attributes", "failure due to invalid attributes"],
    )
    def test__attribute_validation(
        self, item: AnyStr, options: List[AnyStr], expected_output: Optional[Exception]
    ):
        if isinstance(expected_output, type) and issubclass(expected_output, Exception):
            with pytest.raises(expected_output):
                _ = metadata._attribute_validation(item, options)
        else:
            _ = metadata._attribute_validation(item, options)


class TestScript:
    script_obj: metadata.Script
    file: Optional[str]
    directory: Optional[str]
    display: Optional[bool]
    save_images: Optional[bool]
    cutoffs: Optional[List[cons.Cutoffs]] = [
        cons.Cutoffs.BRIGHTNESS,
        cons.Cutoffs.LENGTH,
        cons.Cutoffs.DIFFUSION,
    ]
    brightness_method: Optional[cons.CutoffMethods]
    min_length: Optional[int]
    # diffusion_method: str
    brightness_cutoffs: Optional[Tuple[float, float]]
    diffusion_cutoffs: Optional[Tuple[float, float]]

    @pytest.fixture(autouse=True)
    def setup_and_tear_down(self):
        self.empty_csv_object = metadata.Script(filetype=cons.FileTypes.CSV)
        self.empty_xml_object = metadata.Script(filetype=cons.FileTypes.XML)

    @pytest.mark.parametrize(
        "cutoffs, expected_output",
        [(("brightness", "length"), None), (("brightness", "not_a_cutoff"), KeyError)],
        ids=["successful cutoff validation", "failed cutoff validation"],
    )
    def test__validate_cutoffs(self, cutoffs: List[str], expected_output: Optional[Exception]):
        with patch.object(metadata.Script, "cutoffs", new_callable=PropertyMock) as mock:
            mock.return_value = cutoffs
            if isinstance(expected_output, type) and issubclass(expected_output, Exception):
                with pytest.raises(expected_output):
                    self.empty_csv_object._validate_cutoffs()
            else:
                self.empty_csv_object._validate_cutoffs()

    @pytest.mark.parametrize(
        "filetype, cutoff_method, expected_output",
        [
            (cons.FileTypes.CSV, cons.CutoffMethods.AUTO, None),
            (cons.FileTypes.CSV, "not_a_method", KeyError),
            (cons.FileTypes.CSV, None, None),
            (cons.FileTypes.XML, cons.CutoffMethods.SEMI_AUTO, ValueError),
        ],
        ids=[
            "successful cutoff validation",
            "failed cutoff validation",
            "no cutoffs are fine",
            "missing cutoff values",
        ],
    )
    def test__validate_brightness_method(
        self,
        filetype: cons.FileTypes,
        cutoff_method: Union[cons.CutoffMethods, str],
        expected_output: Optional[Exception],
    ):
        with patch.object(metadata.Script, "brightness_method", new_callable=PropertyMock) as mock:
            mock.return_value = cutoff_method
            if filetype == cons.FileTypes.XML and cutoff_method == cons.CutoffMethods.SEMI_AUTO:
                with pytest.raises(ValueError):
                    self.empty_xml_object._validate_brightness_method()
            else:
                if isinstance(expected_output, type) and issubclass(expected_output, Exception):
                    with pytest.raises(expected_output):
                        self.empty_csv_object._validate_brightness_method()
                else:
                    self.empty_csv_object._validate_brightness_method()
