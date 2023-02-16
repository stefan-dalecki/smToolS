"""Biophysical kinetics"""
from collections import defaultdict
from typing import Self

import numpy as np
import pandas as pd
from scipy import stats

from smToolS import metadata
from smToolS.analysis_tools import formulas as fo
from smToolS.sm_helpers import constants as cons


class Kinetic:
    """Standardized kinetic object"""

    def __init__(self) -> None:
        self.name = None
        self.unit = None
        self.x_label = None
        self.y_label = None
        self.table = None

    def __str__(self) -> str:
        """Retrieve overview of kinetic object

        Returns:
            str: string of object attributes
        """
        return str(self.__dict__)


class Director:
    """Kinetic object constructor"""

    def __init__(self, builder: object) -> None:
        """Initialize the director

        Args:
            builder (object): builds the kinetic class object
        """
        self._builder = builder

    def construct_kinetic(self) -> Self:
        """Calls functions to build each kinetic

        Returns:
            self: class object
        """
        self._builder.generate()
        self._builder.add_attributes()
        self._builder.add_labels()
        self._builder.dataformat()
        return self

    def get_kinetic(self) -> object:
        """Retrieves created class object

        Returns:
            self: kinetic class object
        """
        return self._builder.kinetic


class KineticBuilder:
    """Builds the kinetic"""

    def __init__(self) -> None:
        """Initialize a blank kinetic"""
        self.kinetic = None

    def generate(self) -> None:
        """create an empty kinetic object"""
        self.kinetic = Kinetic()


class BSL(KineticBuilder):
    """Bound State Lifetime

    Args:
        KineticBuilder (class object): inherits from the kinetic builder object
    """

    def __init__(self, script: metadata.Script, df: pd.DataFrame) -> None:
        """Initialize the bound state lifetime object

        Args:
            metadata (class object): persistent metadata
            df (pd.DataFrame): trajectory data
        """
        self.script = script
        self._df = df

    def add_attributes(self) -> None:
        """Add name and unit"""
        self.kinetic.name = "BSL"
        self.kinetic.unit = "sec"

    def add_labels(self) -> None:
        """Add axis labels for display"""
        self.kinetic.x_label = f"Frames after {self.script.min_length}"
        self.kinetic.y_label = "Population Remaining (%)"

    def dataformat(self) -> None:
        """Reformat the trajectory dataframe for model fitting"""
        NUMPERTRACKLEN = pd.DataFrame(columns=["Minimum Frames", "% Tracks"])
        TOT_TRACKS = fo.trajectory_count(self._df)
        STEP = 1
        # The first frame is used as a guide for slowly reducing the remaining
        # trajectories that meet hte length criterion
        start = cutofflen = self.script.min_length - STEP
        while True:
            cutofflen += STEP
            tracks = round(
                fo.trajectory_count(
                    self._df.groupby(cons.TRAJECTORY)
                    .filter(lambda x: len(x) >= cutofflen)
                    .reset_index(drop=True)
                )
                / TOT_TRACKS,
                3,
            )
            new_row = pd.DataFrame.from_dict(
                {"Minimum Frames": [cutofflen - start], "% Tracks": [tracks]}
            )
            NUMPERTRACKLEN = pd.concat([NUMPERTRACKLEN, new_row])
            # Stop when no tracks are remaining
            if tracks == 0:
                break
        self.kinetic.table = NUMPERTRACKLEN


class MSD(KineticBuilder):
    """Mean Squared Displacement

    Args:
        KineticBuilder (class object): builds the MSD class object
    """

    def __init__(self, microscope: metadata.Microscope, df: pd.DataFrame) -> None:
        """Initializes the MSD class object
        Removes the first event in each trajectory

        Args:
            metadata (class object): persistent metadata based on microscope qualities
            df (pd.DataFrame): trajectory data
        """
        # Removes the first step in every single trajectory only for diffusion calculations
        for trajectory in df[cons.TRAJECTORY].unique():
            t_rows = df[df[cons.TRAJECTORY] == trajectory]
            df.drop(min(t_rows.index))
            df.reset_index(drop=True)
        self._microscope = microscope
        self._df = df

    def add_attributes(self) -> None:
        """Add name and unit"""
        self.kinetic.name = cons.MSD
        self.kinetic.unit = cons.MSD_W_UNITS

    def add_labels(self) -> None:
        """Add labels for display"""
        self.kinetic.x_label = "Step Length"
        self.kinetic.y_label = f"Distance ({self.kinetic.unit})"

    def dataformat(self) -> None:
        """Format trajectory dataframe for linear fitting"""
        df = self._df
        step_diffusions = (
            df.groupby(cons.TRAJECTORY)[[cons.Coordinates.X, cons.Coordinates.Y]]
            .apply(self._microscope.calc_all_steps_no_min)
            .to_list()
        )
        result = defaultdict(list)
        for traj in step_diffusions:
            for step in traj:
                result[step].extend(traj[step])
        for i in range(1, 9):
            result[i] = np.mean(result[i])
        mean_SDs = pd.DataFrame(result.items(), columns=["Step Length", cons.MSD])
        self.kinetic.table = mean_SDs

    @staticmethod
    def model(df: pd.DataFrame) -> dict:
        """Generates export dictionary

        Args:
            df (pd.DataFrame): two column dataframe

        Returns:
            dict: export dictionary
        """
        assert df.shape[1] == 2, "Dataframe must contain only two columns"
        x_data = df.iloc[:, 0].values.astype(float)
        y_data = df.iloc[:, 1].values.astype(float)
        try:
            slope, _, r2, pe, se = stats.linregress(x_data, y_data)
        except ValueError:
            slope, _, r2, *_ = np.nan, np.nan, np.nan, np.nan, np.nan
        # Convert data into um^2/sec
        export_dict = {cons.MSD_W_UNITS: slope * 10**8, "MSD R\u00b2:": r2}
        return export_dict


class RayD(KineticBuilder):
    """Rayleigh Distribution / Diffusion

    Args:
        KineticBuilder: builds the kinetic
    """

    def __init__(self, microscope: metadata.Microscope, datalist: list) -> None:
        """Initialize Rayleigh Diffusion object

        Args:
            metadata (object): persistent metadata
            datalist (list): one step square displacements
        """
        self.microscope = microscope
        self._datalist = [i ** (1 / 2) * self.microscope.pixel_size for i in datalist]

    def add_attributes(self) -> None:
        """Add name and unit"""
        self.kinetic.name = "RayDifCoef"
        self.kinetic.unit = "\u03bcm\u00b2/sec"

    def add_labels(self) -> None:
        """Add labels for display"""
        self.kinetic.x_label = f"Diffusion Coefficients ({self.kinetic.unit})"
        self.kinetic.y_label = "Frequency (#)"

    def dataformat(self) -> None:
        """Bin data for Rayleigh Distribution model fitting"""
        sep = 300
        list_max = np.max(self._datalist)
        assert np.min(self._datalist) >= 0, "Diffusion cannot be negative"
        # Separates data between '0' and max values
        bins = np.linspace(0, list_max + (list_max / sep), sep)
        df = pd.DataFrame(self._datalist, columns=["values"])
        bin_data = pd.DataFrame(
            df.groupby(pd.cut(df["values"], bins=bins)).size().values,
            columns=["Frequency (#)"],
        )
        bin_data.index = bins[:-1]
        bin_data = bin_data.reset_index(drop=False)
        self.kinetic.table = bin_data
