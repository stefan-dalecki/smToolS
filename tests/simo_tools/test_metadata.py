import pandas as pd
import pytest
from pandas.testing import assert_frame_equal, assert_series_equal

from simo_tools import metadata as meta


class TestFrame:
    """
    `meta.Frame`
    """

    test_class = meta.Frame

    @pytest.fixture(scope="class")
    def test_obj(self) -> meta.Frame:
        """
        Initialiazed `meta.Frame`.
        """
        return self.test_class(x=1, y=2, brightness=3, frame=4)

    @pytest.fixture(scope="class")
    def test_obj_as_series(self) -> pd.Series:
        """
        `meta.Frame` as a pandas Series.
        """
        return pd.Series(data={"x": 1, "y": 2, "brightness": 3, "frame": 4})

    def test_from_series(self, test_obj: meta.Frame, test_obj_as_series: pd.Series):
        """
        `meta.Frame.from_series`
        """
        assert self.test_class.from_series(test_obj_as_series) == test_obj

    def test_to_series(self, test_obj: meta.Frame, test_obj_as_series: pd.Series):
        """
        `meta.Frame.to_series`
        """
        assert_series_equal(test_obj.to_series(), test_obj_as_series)


class TestTrajectory:
    """
    `meta.Trajectory`
    """

    test_class = meta.Trajectory

    @pytest.fixture(scope="function")
    def test_obj(self) -> meta.Trajectory:
        """
        Initialized `meta.Trajectory`
        """
        return self.test_class(
            id=1,
            frames=[
                meta.Frame(x=1, y=6, brightness=3, frame=4),
                meta.Frame(x=3, y=5, brightness=7, frame=5),
                meta.Frame(x=3, y=8, brightness=4, frame=6),
                meta.Frame(x=4, y=9, brightness=6, frame=7),
                meta.Frame(x=5, y=10, brightness=7, frame=8),
            ],
        )

    @pytest.fixture(scope="function")
    def test_obj_as_df(self) -> pd.DataFrame:
        """
        `meta.Trajectory` as a pandas dataframe.
        """
        return pd.DataFrame(
            data={
                "x": [1, 3, 3, 4, 5],
                "y": [6, 5, 8, 9, 10],
                "brightness": [3, 7, 4, 6, 7],
                "frame": [4, 5, 6, 7, 8],
                "trajectory": 1,
            }
        )

    def test_from_df(self, test_obj: meta.Trajectory, test_obj_as_df: pd.DataFrame):
        """
        `meta.Trajectory.from_df`
        """
        assert self.test_class.from_df(test_obj_as_df) == test_obj

    def test_to_series(self, test_obj: meta.Trajectory, test_obj_as_df: pd.DataFrame):
        """
        `meta.Trajectory.to_df`
        """
        assert_frame_equal(test_obj.to_df(), test_obj_as_df)

    def test___len__(self, test_obj: meta.Trajectory):
        """
        `len(meta.Trajectory)` Length is equal to number of frames.
        """
        assert len(test_obj) == 5

    def test_length(self, test_obj: meta.Trajectory):
        """
        `meta.Trajectory.length Length is equal to number of frames.
        """
        assert test_obj.length == 5

    def test_mean_brightess(self, test_obj: meta.Trajectory):
        """
        `meta.Trajectory.mean_brightness`
        """
        assert test_obj.mean_brightness == 5.4  # (3+7+4+6+7) / 5

    @pytest.mark.parametrize(
        "remove_first_step, expected_output",
        [
            pytest.param(True, 1.9428090415820634, id="remove-first-step"),
            pytest.param(False, 2.016123775561495, id="do-not-remove-first-step"),
        ],
    )
    def test_mean_squared_displacement(
        self, test_obj: meta.Trajectory, remove_first_step: bool, expected_output: float
    ):
        """
        `meta.Trajectory.mean_squared_displacement`
        """
        # No calculations are done, MSD values should be empty
        if remove_first_step:
            assert not test_obj._msd_rm_first
        else:  # `remove_first_step` == `False`
            assert not test_obj._msd_w_first

        assert test_obj.mean_squared_displacement(remove_first_step) == expected_output

        # Calculations are complete, values should be cached
        if remove_first_step:
            assert test_obj._msd_rm_first == expected_output
        else:  # `remove_first_step` == `False`
            assert test_obj._msd_w_first == expected_output
