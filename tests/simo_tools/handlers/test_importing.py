import pandas as pd
from pandas.testing import assert_frame_equal

from simo_tools import constants as cons
from simo_tools.handlers import importing


class TestFileReader:
    test_class = importing.DataFiles

    # @pytest.mark.parametrize(
    #     "path, is_valid",
    #     [
    #         pytest.param(, True, id="valid-file"),
    #         pytest.param("path/to/file/file_name.csv", True, id="valid-dir"),
    #         pytest.param("invalid_location", False, id="invalid-path"),
    #     ],
    # )
    # def test_from_path(self, path: str, is_valid: bool):
    #     """importing.FileReader.__init__
    #     Ensures `TypeError` raised when given invalid path
    #     """
    #     if not is_valid:
    #         with pytest.raises(TypeError):
    #             _ = self.test_class.from_path(path, cons.ReadFileTypes.CSV)
    #     else:
    #         _ = self.test_class.from_path(path, cons.ReadFileTypes.CSV)

    def test__import_csv(
        self, raw_trajectories_csv_path: str, raw_trajectories_csv_df: pd.DataFrame
    ):
        """
        imporing.FileReader._import_csv.
        """
        test_obj = self.test_class(
            filepaths=set(raw_trajectories_csv_path), filetype=cons.ReadFileTypes.CSV
        )
        actual_output = test_obj._import_csv(filepath=raw_trajectories_csv_path)
        raw_trajectories_csv_df.rename(columns={cons.M2: cons.BRIGHTNESS}, inplace=True)
        raw_trajectories_csv_df.columns = raw_trajectories_csv_df.columns.str.lower()
        assert_frame_equal(actual_output.df, raw_trajectories_csv_df)
