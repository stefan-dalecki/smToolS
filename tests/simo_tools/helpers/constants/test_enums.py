import pytest

from simo_tools.helpers.constants import enums


class MockEnum(enums.EnumHelper):
    KEY_1 = "value_1"
    KEY_2 = "value_2"
    KEY_3 = "value_3"


class TestEnumHelper:
    """
    Tests `EnumHelper` class.
    """

    test_class = enums.EnumHelper

    @pytest.fixture(scope="class")
    def mock_enum(self):
        return MockEnum

    def test_list_of_options(self, mock_enum: MockEnum):
        """
        EnumHelper.list_of_options()
        """
        assert mock_enum.list_of_options() == ["value_1", "value_2", "value_3"]

    def test_set_of_options(self, mock_enum: MockEnum):
        """
        EnumHelper.set_of_options()
        """
        assert mock_enum.set_of_options() == {"value_1", "value_2", "value_3"}
