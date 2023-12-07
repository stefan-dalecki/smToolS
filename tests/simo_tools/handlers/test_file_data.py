from typing import Optional, cast

import pytest

from simo_tools import constants as cons
from simo_tools import metadata as meta
from simo_tools.handlers import file_data


@pytest.mark.parametrize(
    "filetype, error",
    [
        pytest.param(".csv", None, id="valid-filetype"),
        pytest.param(".jpg", ValueError, id="invalid-filetype"),
    ],
)
def test__validate_filetype(filetype: str, error: Optional[ValueError]):
    """
    file_data._validate_filetype.
    """
    if error:
        with pytest.raises(error):
            file_data._validate_filetype(filetype)
    else:
        assert filetype.strip(".") == file_data._validate_filetype(filetype)


class TestFileData:
    test_class = file_data.FileData

    @pytest.fixture(scope="class")
    def test_obj(self) -> file_data.FileData:
        """
        Initialized test class.
        """
        return self.test_class(
            filetype=cons.ReadFileTypes.CSV, movies=[cast(meta.Movie, "placeholder")]
        )

    def test_set_units(self, test_obj):
        """
        file_data.FileData.set_units.
        """
        assert not test_obj.pixel_size and not test_obj.fps
        test_obj.set_units(pixel_size=123, fps=456)
        assert test_obj.pixel_size == 123 and test_obj.fps == 456
