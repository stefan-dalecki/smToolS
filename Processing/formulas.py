"""Various function shortcuts"""

from functools import reduce
from collections import defaultdict
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

    @staticmethod
    def all_steps(df: pd.DataFrame, metadata: object) -> defaultdict(list):
        """Calculate MSD for 8 step lengths

        Args:
            df (pd.DataFrame): trajectory data

        Returns:
            defaultdict(list): squared distance for each step length
        """
        df = df.reset_index(drop=True)
        x_col, y_col = df["x"], df["y"]
        all_steps = defaultdict(list)
        if len(df) - 3 > 8:
            max_step_len = 8
        else:
            max_step_len = len(df) - 3
        for step_len in range(1, max_step_len + 1):
            for step_num in range(0, len(df) - step_len - 1):
                x1, y1 = x_col[step_num], y_col[step_num]
                x2, y2 = x_col[step_num + step_len], y_col[step_num + step_len]
                distance = (
                    Calc.distance(x1, y1, x2, y2) ** 2
                    * metadata.pixel_size**2
                    / (4 * metadata.framestep_size)
                )
                all_steps[step_len].append(distance)
        return all_steps

    @staticmethod
    def e_estimation(df: pd.DataFrame) -> float:
        """Estimate decay time constant

        Args:
            df (pd.DataFrame): formatted BSL kinetic table

        Returns:
            float: 1/e estimation of time constant
        """

        estimation = df.iloc[(df.iloc[:, 1] - (1 / np.exp(1))).abs().argsort()[:1]]
        estimation = estimation.iloc[:, 0].values[0]
        return estimation

    @staticmethod
    def f_modeltest(ss1: float, ss2: float, degfre1: int, degfre2: int) -> float:
        """Generate f-statistic comparing two models

        Args:
            ss1 (float): sums squared
            ss2 (flaot): sums square 2
            degfre1 (int): degrees of freedom
            degfre2 (int): degrees of freedom 2

        Returns:
            float: f-statistic
        """
        assert degfre1 > degfre2, "Simpler model is after the complex model"
        f_stat = ((ss1 - ss2)(degfre1 - degfre2)) / (ss2 / degfre2)
        return f_stat

    @staticmethod
    def depth_of_field(lam=532, incidence=69, n2=1.33, n1=1.515) -> float:
        """smTIRF depth of field

        Args:
            lam (int, optional): laser wavelength. Defaults to 532.
            incidence (int, optional): incidence. Defaults to 69.
            n2 (float, optional): refractive index 2. Defaults to 1.33.
            n1 (float, optional): refractive index. Defaults to 1.515.

        Returns:
            float: microscope depth of field
        """

        theta = incidence * (180 / np.pi)
        depth_of_field = (lam / (4 * np.pi)) * (
            n1**2 * (np.sin(theta)) ** 2 - n2**2
        ) ** (-1 / 2)
        return depth_of_field

    @staticmethod
    def luminescence(I0: float, d: float, z: float) -> float:
        """Intensity at distances off of coverslip

        Args:
            I0 (float): initial intensity
            d (float): depth of field
            z (float): distance from coverslip

        Returns:
            float: intensity at z-distance
        """

        Iz = I0 ** (-z / d)
        return Iz


class Find:
    """Find subset of larger group"""

    @staticmethod
    def identifiers(
        full_string: str,
        separator: str,
        value_name: str,
        value_search_names: list,
        *,
        failure: str = "not found",
    ) -> dict:
        """Find image attributes from file name

        Args:
            full_string (str): full string in question
            separator (str): separator between sub string elements
            value_name (str): value of interest
            value_search_names (list): potential value names
            failure (str, optional): what to return if nothing is found. Defaults to "not found".

        Raises:
            RuntimeError: nothing is found

        Returns:
            dict: search result as dictionary
        """

        output = {}
        separated_full_string = [i for i in full_string.lower().split(separator)]
        for search_name in value_search_names:
            hit_indeces = [
                i
                for i, val in enumerate(separated_full_string)
                if val.find(search_name) != -1
            ]
            for hit_index in hit_indeces:
                if "-" in separated_full_string[hit_index]:
                    if "+" in separated_full_string[hit_index]:
                        protein_descriptions = separated_full_string[hit_index].split(
                            "+"
                        )
                    else:
                        protein_descriptions = [separated_full_string[hit_index]]
                    for protein_description in protein_descriptions:
                        concentration, protein = protein_description.split("-")
                        protein = protein.upper()
                        concentration_value = concentration[:-2]
                        concentration_units = concentration[-2:]
                        output[
                            f"{protein} ({concentration_units})"
                        ] = concentration_value
                else:
                    output[value_name] = separated_full_string[hit_index].replace(
                        str(search_name), ""
                    )
                    break
        if not output:
            output[value_name] = failure
        return output


class Form:
    """Reform already generated data"""

    @staticmethod
    def reorder(df: pd.DataFrame, column_name: str, location: int) -> pd.DataFrame:
        """Reorder dataframe columns

        Args:
            df (pd.DataFrame): dataframe
            column_name (str): name of column to move
            location (int): new column location index

        Returns:
            pd.DataFrame: rearrange dataframe
        """

        assert column_name in df.columns, "Column not found"
        column_data = df[column_name].values
        df = df.drop(columns=[column_name])
        df.insert(location, column_name, column_data)
        return df

    @staticmethod
    def userinput(search_string: str, option_list: list) -> str:
        """Allow user to choose from options

        Args:
            search_string (str): option choice
            option_list (list): list of options

        Returns:
            str: user choice
        """

        print(f"\nAvailable options\n{option_list}", end="\n" * 2)
        while True:
            ans = input(f"From options above, select your {search_string}: ")
            if ans not in option_list:
                print("Not an available option\n")
            else:
                break
        return ans

    @staticmethod
    def inputbool(input_string: str) -> bool:
        """User input that results in true or false

        Args:
            input_string (str): question for user

        Returns:
            bool: binary decision
        """

        while True:
            ans = input(f"\n{input_string} (y/n): ")
            if ans == "y":
                ans = True
                break
            elif ans == "n":
                ans = False
                break
            else:
                print("Not an available option\n")
        return ans

    @staticmethod
    def ordinal(number: int) -> str:
        """Convert integer to ordinal string

        Args:
            number (int): integer

        Returns:
            str: ordinal string (i.e. 1st, 2nd, 89th, ...)
        """
        return "%d%s" % (
            number,
            "tsnrhtdd"[
                (number / 10 % 10 != 1) * (number % 10 < 4) * (number % 10) :: 4
            ],
        )
