"""Biophysical kinetics"""

from collections import defaultdict
import pandas as pd
import numpy as np
from scipy import stats
import formulas as fo


class Kinetic:
    """Standardized kinetic object"""

    def __init__(self) -> None:
        self.name = None
        self.unit = None
        self.x_label = None
        self.y_label = None
        self.table = None

    def __str__(self) -> None:
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

    def construct_kinetic(self) -> None:
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

    def __init__(self, metadata: object, df: pd.DataFrame) -> None:
        """Initialize the bound state lifetime object

        Args:
            metadata (class object): persistent metadata
            df (pd.DataFrame): trajectory data
        """
        self.metadata = metadata
        self._df = df

    def add_attributes(self) -> None:
        """Add name and unit"""
        self.kinetic.name = "BSL"
        self.kinetic.unit = "sec"

    def add_labels(self) -> None:
        """Add axis labels for display"""
        self.kinetic.x_label = f"Frames after {self.metadata.frame_cutoff}"
        self.kinetic.y_label = "Population Remaining (%)"

    def dataformat(self) -> None:
        """Reformat the trajectory dataframe for model fitting"""
        numpertracklen = pd.DataFrame(columns=["Minimum Frames", "% Tracks"])
        tot_tracks = fo.Calc.trajectory_count(self._df)
        step = 2
        start = cutofflen = self.metadata.frame_cutoff - 2
        while True:
            cutofflen += step
            tracks = round(
                fo.Calc.trajectory_count(
                    self._df.groupby("Trajectory")
                    .filter(lambda x: len(x) >= cutofflen)
                    .reset_index(drop=True)
                )
                / tot_tracks,
                3,
            )
            new_row = pd.DataFrame.from_dict(
                {"Minimum Frames": [cutofflen - start], "% Tracks": [tracks]}
            )
            numpertracklen = pd.concat([numpertracklen, new_row])
            if tracks == 0:
                break
        self.kinetic.table = numpertracklen


class MSD(KineticBuilder):
    """Mean Squared Displacement

    Args:
        KineticBuilder (class object): builds the MSD class object
    """

    def __init__(self, metadata: object, df: pd.DataFrame) -> None:
        """Initializes the MSD class object
        Removes the first event in each trajectory

        Args:
            metadata (class object): persistent metadata based on microscope qualities
            df (pd.DataFrame): trajectory data
        """
        for trajectory in df["Trajectory"].unique():
            t_rows = df[df["Trajectory"] == trajectory]
            df.drop(min(t_rows.index))
            df.reset_index(drop=True)
        self.metadata = metadata
        self._df = df

    def add_attributes(self) -> None:
        """Add name and unit"""
        self.kinetic.name = "MSD"
        self.kinetic.unit = "\u03BCm\u00b2/sec"

    def add_labels(self) -> None:
        """Add labels for display"""
        self.kinetic.x_label = "Step Length"
        self.kinetic.y_label = f"Distance ({self.kinetic.unit})"

    def dataformat(self) -> None:
        """Format trajectory dataframe for linear fitting"""
        df = self._df
        step_diffusions = (
            df.groupby("Trajectory")[["x", "y"]]
            .apply(fo.Calc.all_steps, self.metadata)
            .to_list()
        )
        result = defaultdict(list)
        for traj in step_diffusions:
            for step in traj:
                result[step].extend(traj[step])
        for i in range(1, 9):
            result[i] = np.mean(result[i])
        mean_SDs = pd.DataFrame(result.items(), columns=["Step Length", "MSD"])
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
            slope, intercept, r2, p, se = stats.linregress(x_data, y_data)
        except ValueError:
            slope, intercept, r2, pe, se = np.nan, np.nan, np.nan
        export_dict = {"MSD (\u03BCm\u00b2/sec)": slope * 10**8, "MSD R\u00b2:": r2}
        return export_dict


class RayD(KineticBuilder):
    """Rayleigh Distribution / Diffusion

    Args:
        KineticBuilder: builds the kinetic
    """

    def __init__(self, metadata: object, datalist: list) -> None:
        """Initialize Rayleigh Diffusion object

        Args:
            metadata (object): persistent metadata
            datalist (list): one step square displacements
        """
        self.metadata = metadata
        self._datalist = [i ** (1 / 2) * self.metadata.pixel_size for i in datalist]

    def add_attributes(self) -> None:
        """Add name and unit"""
        self.kinetic.name = "RayDifCoef"
        self.kinetic.unit = "\u03BCm\u00b2/sec"

    def add_labels(self) -> None:
        """Add labels for display"""
        self.kinetic.x_label = f"Diffusion Coefficients ({self.kinetic.unit})"
        self.kinetic.y_label = "Frequency (#)"

    def dataformat(self) -> None:
        """Bin data for Rayleigh Distribution model fitting"""
        sep = 300
        list_max = np.max(self._datalist)
        bins = np.linspace(0, list_max + (list_max / sep), sep)
        df = pd.DataFrame(self._datalist, columns=["values"])
        bin_data = pd.DataFrame(
            df.groupby(pd.cut(df["values"], bins=bins)).size().values,
            columns=["Frequency (#)"],
        )
        bin_data.index = bins[:-1]
        bin_data = bin_data.reset_index(drop=False)
        self.kinetic.table = bin_data
