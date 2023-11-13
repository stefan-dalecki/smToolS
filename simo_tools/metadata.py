from copy import copy
from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Self

import numpy as np
import pandas as pd

import simo_tools.constants as cons
from simo_tools.analysis import formulas


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
    table: pd.DataFrame
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


@dataclass
class MovieMeta:
    """
    Initialize microscope parameters.

    These values are dependent on the qualities of your microscope/movie
    Args:
        pixel_size (float): pixel width in cm. Defaults to 0.000024.
        framestep_size (float): time between frames. Defaults to 0.0217.

    """

    pixel_size: float
    framestep_size: float

    def modify(self, **kwargs: dict[str, Any]) -> Self:
        """
        Temporarily modify metadata.

        Useful if you want to run different sub-routines within your program that
        change the frame_cutoff or other characteristic
        Returns:
            object: modified metadata

        """
        temporary_metadata = copy(self)
        for key, val in kwargs.items():
            setattr(temporary_metadata, key, val)
        return temporary_metadata


@dataclass
class Movie(MovieMeta):
    trajectories: Trajectories
