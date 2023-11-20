import pytest

from simo_tools.constants import enums

VALUE_1 = "value_1"
VALUE_2 = "value_2"
VALUE_3 = "value_3"


class MockEnum(enums.EnumHelper):
    KEY_1 = VALUE_1
    KEY_2 = VALUE_2
    KEY_3 = VALUE_3


class TestEnumHelper:
    """
    Tests `EnumHelper` class.
    """

    test_class = MockEnum

    def test_list_of_options(self):
        """
        EnumHelper.list_of_options()
        """
        assert self.test_class.list_of_options() == [VALUE_1, VALUE_2, VALUE_3]

    def test_set_of_options(self):
        """
        EnumHelper.set_of_options()
        """
        assert self.test_class.set_of_options() == {VALUE_1, VALUE_2, VALUE_3}

    @pytest.mark.parametrize(
        "key, error",
        [
            pytest.param("KEY_1", False, id="matches-case"),
            pytest.param("key_1", False, id="lower-case"),
            pytest.param("not_key", True, id="invalid-key"),
        ],
    )
    def test___getitem__(self, key: str, error: bool):
        """
        EnumHelper.__getitem__
        """
        if error:
            with pytest.raises(KeyError):
                _ = self.test_class[key]
        else:
            _ = self.test_class[key]

    def test_enum_meta(self):
        """
        EnumHelperMeta.__getitem__
        """
        assert self.test_class[".key_1"] == self.test_class["KEY_1"]
