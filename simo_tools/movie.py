from dataclasses import dataclass

import pandas as pd

from simo_tools import generic_funcs as gf
from simo_tools import trajectory as traj

# from simo_tools.analysis import kinetics as kins
from simo_tools.handlers import importing


@dataclass(frozen=True)
class MovieParams:
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
class Movie:
    """
    Single file converted into `Trajectories`.
    """

    path: str
    trajectories: traj.Trajectories
    # kinetics: Optional[list[type[kins.Kinetic]]] = None

    @classmethod
    def from_path(cls, path: str):
        """
        Infers filetype from path and imports table before converting to `Trajectories`.
        """
        df = cls.import_file(path)
        return cls(path=path, trajectories=traj.Trajectories.from_df(df))

    @staticmethod
    def import_file(path: str) -> pd.DataFrame:
        """
        Reads in dataframe from path.
        """
        return importing.import_table(path)

    def __repr__(self):
        return gf.build_repr(self)

    def __len__(self):
        return len(self.trajectories)

    @property
    def thresholded_trajectories(self) -> traj.Trajectories:
        """
        Trajectories which have passed cuttofs determined through
        `data.DataFiles.apply_cutoffs.
        """
        return self._thresholded_trajectories

    @thresholded_trajectories.setter
    def thresholded_trajectories(self, thresholded_trajectories: traj.Trajectories):
        self._thresholded_trajectories = thresholded_trajectories

    @property
    def analysis_trajectories(self) -> traj.Trajectories:
        """
        If thresholded trajectories are available, use those.

        If trajectories have not yet been thresholded, return the
        originals.

        """
        return self.thresholded_trajectories or self.trajectories

    # def analyze(self, pixel_size: float, fps: float):
    #     """Generate kinetics and fit with built-in models"""
    #     analyzer = analyzing.Analyzer(self, pixel_size, fps)
    #     analyzer.construct_kinetics()
    #     return "something"
