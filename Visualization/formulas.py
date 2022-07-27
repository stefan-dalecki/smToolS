"""Various function shortcuts"""

from functools import reduce
import operator
import numpy as np
import pandas as pd


class Calc:
    """Basic formulas to speed up calculations"""

    @staticmethod
    def trio(metadata: object, df: pd.DataFrame) -> pd.DataFrame:
        """Calculate average brightness, length, and diffusivity for each trajectory

        Args:
            df (pd.DataFrame): trajectory data

        Returns:
            pd.DataFrame: updated trajectory data with trio
        """
        df["Average_Brightness"] = df.groupby("Trajectory")["Brightness"].transform(
            np.mean
        )
        df["Length (frames)"] = df.groupby("Trajectory")["Trajectory"].transform("size")
        data = df.groupby("Trajectory")[["x", "y"]].apply(Calc.one_step_MSD, metadata)
        data = pd.DataFrame(data.to_list(), columns=["SDs", "MSD"])
        data.index += 1
        data.reset_index(inplace=True)
        data = data.rename(columns={"index": "Trajectory"})
        df["SDs"] = reduce(operator.add, data["SDs"])
        df = df.merge(data[["Trajectory", "MSD"]])
        return df

    @staticmethod
    def trajectory_count(df: pd.DataFrame) -> int:
        """Counts numbers of trajectories remaining

        Args:
            df (pd.DataFrame): trajectory data

        Returns:
            int: number of trajectories
        """

        trajectory_count = df["Trajectory"].nunique()
        return trajectory_count

    @staticmethod
    def distance(x1: float, y1: float, x2: float, y2: float) -> float:
        """Calulate distance between two coordinates

        Args:
            x1 (float): x-value
            y1 (float): y-value
            x2 (float): x2-value
            y2 (float): y2-value

        Returns:
            float: distance between two coordinates
        """

        distance = (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** (1 / 2)
        return distance

    @staticmethod
    def one_step_MSD(df: pd.DataFrame, metadata: object) -> tuple:
        """Calculate preliminary mean squared displacement

        Args:
            df (pd.DataFrame): data for one trajectory

        Returns:
            float: mean squared displacement
        """
        df = df.reset_index(drop=True)
        x_col, y_col = df["x"], df["y"]
        if len(df) > 1:
            SDs = [np.nan]
            for i in range(len(df) - 1):
                x1, y1 = x_col[i], y_col[i]
                x2, y2 = x_col[i + 1], y_col[i + 1]
                squared_distance = Calc.distance(x1, y1, x2, y2) ** 2
                SDs.append(squared_distance)
            MSD = (
                np.nanmean(SDs)
                * metadata.pixel_size**2
                / (4 * metadata.framestep_size)
                * 1e8
            )
            return SDs, MSD
        else:
            return 0, 0
