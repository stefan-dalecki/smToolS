from dataclasses import dataclass
from typing import Literal, Optional, Self

import numpy as np
import pandas as pd

from simo_tools import constants as cons
from simo_tools import generic_funcs as gf

COLUMNS = cons.BASE_TRAJECTORY_TABLE_COLS


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

    def to_series(self):
        """
        Converts object to a pandas Series.
        """
        return pd.Series(self.__dict__)


@dataclass(unsafe_hash=True)
class Trajectory:
    """
    One trajectory with frame data.
    """

    id: int  # noqa: A003
    frames: list[Frame]
    _length: Optional[int] = None  # set via property retrieval
    _mean_brightness: Optional[float] = None  # set via property retrieval
    _msd_w_first: Optional[float] = None  # set via property retrieval
    _msd_rm_first: Optional[float] = None  # set via property retrieval

    @classmethod
    def from_df(cls, df: pd.DataFrame):
        """
        Each row consists of one frame.
        """
        id = df[cons.TRAJECTORY].unique()[0]  # noqa: A001
        df.drop(columns=cons.TRAJECTORY, inplace=True)
        return cls(id=id, frames=df.apply(Frame.from_series, axis=1).to_list())

    def to_df(self) -> pd.DataFrame:
        """
        Converts object ot a pandas DataFrame.
        """
        df = pd.DataFrame([frame.to_series() for frame in self.frames])
        df["trajectory"] = self.id
        return df

    def __len__(self):
        """
        Number of total frames.
        """
        return len(self.frames)

    def __repr__(self):
        return gf.build_repr(self, ignore_attrs=["frames"])

    @property
    def length(self):
        """
        Number of frames.
        """
        if self._length:
            return self._length
        self._length = len(self)
        return self._length

    @property
    def mean_brightness(self):
        """
        Average brightness of all frames.
        """
        if self._mean_brightness:
            return self._mean_brightness
        self._mean_brightness = np.mean([frame.brightness for frame in self.frames])
        return self._mean_brightness

    def mean_squared_displacement(self, remove_first_step: bool = True) -> float:
        """
        Mean Squared Displacement (MSD).

        Caches MSD calculation based on whether we keep or remove the first frame step.

        Args:
            remove_first_step (bool, optional): First frame is often very noisy.
                                                Defaults to True.

        Returns:
            float: mean squared displacement in microns squared per second

        """
        target_msd_step = self._check_msd_exists(remove_first_step)
        if target_msd_step:
            return target_msd_step

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
        msd = cast(float, np.mean(all_displacements))
        return self._set_and_return_msd_attr(msd, remove_first_step)

    def _check_msd_exists(self, remove_first_step: bool) -> Optional[float]:
        """
        Check if we have calculated `_mean_squared_displacement` with or without a first
        trajectory frame.

        Args:
            remove_first_step (bool): True if we removed the first frame,
                                        False if using all frames.

        Returns:
            Optional[float]: MSD value if it has been calculated for given
                                `remove_first_step` criterion

        """
        return self._msd_w_first if remove_first_step else self._msd_rm_first

    def _set_and_return_msd_attr(self, val: float, remove_first_step: bool):
        """
        Sets and returns `_msd_rm_first` or `_msd_w_first` based on `remove_first_step`.

        Args:
            val (float): MSD
            remove_first_step (bool): True if we removed the first frame,
                                        False if using all frames.
        Returns:
            float: mean squared displacement in microns squared per second

        """
        MSD_ATTR_MAP = {True: "_msd_rm_first", False: "_msd_w_first"}
        attr = MSD_ATTR_MAP[remove_first_step]
        setattr(self, attr, val)
        return getattr(self, attr)


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

    @property
    def length(self) -> list[float]:
        """
        Number of frames for each trajectory.
        """
        return [traj.length for traj in self]

    @property
    def mean_brightness(self) -> list[float]:
        """
        Average brightness of all frames for each trajectory.
        """
        return [traj.mean_brightness for traj in self]


def get_trajectories(
    trajs_1: Trajectories, trajs_2: Trajectories, method: Literal["shared", "unique"]
) -> Trajectories:
    """
    Compares two `Trajectories` objects and returns only trajectories that are shared
    between the two.
    """
    METHOD_OPTIONS = ["shared", "unique"]
    if method == "shared":
        return Trajectories([traj for traj in trajs_1 if traj.id in trajs_2.ids])
    if method == "unique":
        return Trajectories([traj for traj in trajs_1 if traj.id not in trajs_2.ids])
    raise ValueError(f"Method must be one of: `{', '.join(METHOD_OPTIONS)}`.")
