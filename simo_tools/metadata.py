from dataclasses import dataclass
from itertools import pairwise
from typing import Self

import numpy as np
import pandas as pd

import simo_tools.constants as cons
from simo_tools.analysis import formulas

__all__ = [
    "Meta",
    "Trajectories",
]


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
        return cls.from_dict(dict(series))

    @classmethod
    def from_dict(cls, dictionary: dict) -> Self:
        """
        Dict keys must match class keys.
        """
        return cls(**dictionary)


@dataclass
class Trajectory:
    """
    One trajectory with frame data.
    """

    frames: list[Frame]

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """
        Each row consists of one frame.
        """
        return cls(frames=df.apply(Frame.from_series, axis=1).to_list())

    @property
    def length(self):
        """
        Number of total frames.
        """
        return len(self.frames)

    @property
    def mean_brightness(self):
        """
        Average brightness of all frames.
        """
        return np.mean([frame.brightness for frame in self.frames])

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


@dataclass
class Trajectories:
    """
    Primarily used to avoid working entirely in dataframes.
    """

    COLUMNS = cons.BASE_TRAJECTORY_TABLE_COLS
    trajectories: list[Trajectory]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """
        Uses trajectory data from table to create instance.
        """
        return cls(
            table=df,
            trajectories=df[cls.COLUMNS]
            .groupby(cons.TRAJECTORY)
            .apply(Trajectory.from_df)
            .to_list(),
        )

    @property
    def num(self):
        """
        Number of trajectories.
        """
        return len(self.trajectories)

    @property
    def table(self):
        """
        Trajectories as dataframe.
        """
        pass


@dataclass
class Movie:
    trajectories: Trajectories
