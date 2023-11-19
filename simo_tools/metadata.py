from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Self

import numpy as np
import pandas as pd

import simo_tools.constants as cons
from simo_tools.analysis import formulas
from simo_tools.handlers import importing

__all__ = [
    "Meta",
    "Trajectories",
    "Movie",
]

COLUMNS = cons.BASE_TRAJECTORY_TABLE_COLS


def build_repr(obj: Any, *, ignore_attrs: list[str] = []) -> str:
    """
    Creates string of class name with all attributes.
    """
    attrs = []
    for attr, val in obj.__dict__.items():
        if not attr.startswith("_") and attr not in ignore_attrs:
            attrs += [f"{attr}={val}"]
    return f"{obj.__class__.__name__}({', '.join(attrs)})"


@dataclass(frozen=True)
class Meta:
    """
    Initialize microscope parameters.

    Acts as a singleton throughout the codebase
    These values are dependent on the qualities of your microscope/movie
    Args:
        pixel_size (float): pixel width in cm
        framestep_size (float): time between frames

    """

    pixel_size: float
    framestep_size: float


@dataclass
class Frame:
    """
    Single particle frame.
    """

    x: float
    y: float
    brightness: float
    frame: int

    @classmethod
    def from_series(cls, series: pd.Series) -> Self:
        """
        Series keys must match class keys.

        Convert series to dictionary before using key-value pairs as
        kwargs.

        """
        return cls(**dict(series))


@dataclass
class Trajectory:
    """
    One trajectory with frame data.
    """

    id: int
    frames: list[Frame]

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """
        Each row consists of one frame.
        """
        id = df[cons.TRAJECTORY].unique()[0]
        df.drop(columns=cons.TRAJECTORY, inplace=True)
        return cls(id=id, frames=df.apply(Frame.from_series, axis=1).to_list())

    def __len__(self):
        """
        Number of total frames.
        """
        return len(self.frames)

    def __repr__(self):
        return build_repr(self, ignore_attrs=["frames"])

    @property
    def length(self):
        """
        Number of frames.
        """
        return len(self)

    @property
    def mean_brightness(self):
        """
        Average brightness of all frames.
        """
        return np.mean([frame.brightness for frame in self.frames])

    @property
    def mean_squared_displacement(self, remove_first_step: bool = True) -> float:
        """
        Mean Squared Displacement.

        Args:
            remove_first_step (bool, optional): First frame is often very noisy.
                                                Defaults to True.

        Returns:
            float: mean squared displacement in microns squared per second

        """
        msd_frames = self.frames.copy()
        if remove_first_step:
            msd_frames = msd_frames[1:]

        all_displacements = []

        # iterate through `1` and `2`, `2` and `3`, `3` and `4`, etc...
        for cur_frame, next_frame in pairwise(msd_frames):
            displacement = formulas.calc_distance(
                cur_frame.x, cur_frame.y, next_frame.x, next_frame.y
            )
            all_displacements += [displacement]

        return np.mean(all_displacements)


class Trajectories(list[Trajectory]):
    """
    Primarily used to avoid working entirely in dataframes.
    """

    def __init__(self, trajectories: list[Trajectory]):
        super().__init__(trajectories)

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """
        Uses trajectory data from table to create instance.
        """
        traj_list = (
            df[COLUMNS].groupby(cons.TRAJECTORY).apply(Trajectory.from_df).to_list()
        )
        return cls(traj_list)

    def __repr__(self):
        return f"{self.__class__.__name__}(num={len(self)})"

    @property
    def ids(self) -> list[int]:
        """
        IDs for all trajectories, useful for finding shared trajectories between
        objects.
        """
        return [traj.id for traj in self]


def get_shared_trajectories(
    trajs_1: Trajectories, trajs_2: Trajectories
) -> Trajectories:
    """
    Compares two `Trajectories` objects and returns only trajectories that are shared
    between the two.
    """
    return Trajectories([traj for traj in trajs_1 if traj.id in trajs_2.ids])


@dataclass
class Movie:
    """
    Single file converted into `Trajectories`.
    """

    path: str
    trajectories: Trajectories

    @classmethod
    def from_path(cls, path: str):
        """
        Infers filetype from path and imports table before converting to `Trajectories`.
        """
        df = cls.import_file(path)
        return cls(path=path, trajectories=Trajectories.from_df(df))

    @staticmethod
    def import_file(path: str) -> pd.DataFrame:
        """
        Reads in dataframe from path.
        """
        return importing.import_table(path)

    def __repr__(self):
        return build_repr(self)

    def __len__(self):
        return len(self.trajectories)

    @property
    def thresholded_trajectories(self) -> Trajectories:
        """
        Trajectories which have passed cuttofs determined through
        `data.DataFiles.apply_cutoffs.
        """
        return self._thresholded_trajectories

    @thresholded_trajectories.setter
    def thresholded_trajectories(self, thresholded_trajectories: Trajectories):
        self._thresholded_trajectories = thresholded_trajectories
