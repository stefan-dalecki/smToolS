from dataclasses import dataclass
from itertools import pairwise
from typing import Self

import numpy as np
import pandas as pd

from simo_tools.analysis import formulas


@dataclass
class Frame:
    x: float
    y: float
    brightness: float
    frame: int


@dataclass
class Trajectory:
    """
    One trajectory with frame data.
    """

    frames: list[Frame]

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
            remove_first_step (bool, optional): First trajectory frame is often very noisy. Defaults to True.

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
    trajectories: set[Trajectory]

    @classmethod
    def from_df(cls, df: pd.DataFrame) -> Self:
        """
        Uses trajectory data from table to create instance.
        """
        trajectories = []
        return cls(trajectories)
