import pandas as pd
import pytest

from simo_tools import metadata as meta


class TestFrame:
    """
    trajectory.Frame.
    """

    test_class = meta.Frame

    def test_from_series(self):
        """
        meta.Frame.from_series(...)
        """
        series = pd.Series(data={"x": 1.23, "y": 4.56, "brightness": 789, "frame": 4})
        expected_result = self.test_class(x=1.23, y=4.56, brightness=789, frame=4)
        actual_result = self.test_class.from_series(series)
        assert actual_result == expected_result

    def test_from_dict(self):
        """
        meta.Frame.from_dict(...)
        """
        dictionary = {"x": 1.23, "y": 4.56, "brightness": 789, "frame": 4}
        expected_result = self.test_class(x=1.23, y=4.56, brightness=789, frame=4)
        actual_result = self.test_class.from_dict(dictionary)
        assert actual_result == expected_result


class TestTrajectory:
    test_class = meta.Trajectory

    @pytest.fixture(scope="class")
    def traj_df(self) -> pd.DataFrame:
        """
        Fixture for reuse throughout tests.
        """
        data = {
            "x": [1, 2, 3],
            "y": [4, 5, 6],
            "brightness": [7, 8, 9],
            "frame": [10, 11, 12],
        }
        return pd.DataFrame.from_dict(data)

    @pytest.fixture(scope="class")
    def traj_obj(self) -> meta.Trajectory:
        """
        Fixture for reuse throughout tests.
        """
        list_frames = [
            meta.Frame(x=1.123, y=4.658, brightness=7, frame=10),
            meta.Frame(x=1.143, y=4.148, brightness=8, frame=11),
            meta.Frame(x=2.496, y=5.466, brightness=9, frame=12),
            meta.Frame(x=2.472, y=10.806, brightness=10, frame=13),
            meta.Frame(x=1.621, y=9.901, brightness=11, frame=14),
        ]
        return self.test_class(frames=list_frames)

    def test_from_df(self, traj_df: pd.DataFrame):
        """
        meta.Trajectory.from_df(...)
        """
        actual_result = self.test_class.from_df(traj_df)
        expected_result = self.test_class([
            meta.Frame(x=1, y=4, brightness=7, frame=10),
            meta.Frame(x=2, y=5, brightness=8, frame=11),
            meta.Frame(x=3, y=6, brightness=9, frame=12),
        ])
        assert actual_result == expected_result

    def test_length(self, traj_obj: meta.Trajectory):
        """
        meta.Trajectory.length.
        """
        assert traj_obj.length == 5

    def test_mean_brightness(self, traj_obj: meta.Trajectory):
        """
        meta.Trajectory.mean_brightness.
        """
        assert traj_obj.mean_brightness == 9  # (7+8+9+10+11) / 5

    @pytest.mark.parametrize(
        "remove_first_step, expected_result",
        [
            pytest.param(True, 2.8237215883484033, id="remove-first-step"),
            pytest.param(False, 2.245389192813051, id="do-not-remove-first-step"),
        ],
    )
    def test_mean_square_displacement(
        self, traj_obj: meta.Trajectory, remove_first_step: bool, expected_result: float
    ):
        """
        meta.Trajectory.mean_squared_displacement.
        """

        assert traj_obj.mean_squared_displacement(remove_first_step) == expected_result


class TestTrajectories:
    test_class = meta.Trajectories

    @pytest.fixture(scope="class")
    def trajs_dict(self) -> dict[str, list[int | float]]:
        return {
            "x": [1.123, 1.143, 2.496, 2.472, 1.621],
            "y": [4.658, 4.148, 5.466, 10.806, 9.901],
            "brightess": [7, 8, 9, 10, 11],
            "frame": [10, 11, 12, 13, 14, 15],
        }

    def test_from_df(self, trajectories_df: pd.DataFrame):
        """
        meta.Trajectores.from_df.
        """
        cls_obj = self.test_class.from_df(trajectories_df)
        assert len(cls_obj.trajectories) == 4
