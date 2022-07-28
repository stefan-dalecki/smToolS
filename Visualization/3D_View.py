import os
import sys
import operator
from functools import reduce
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tkinter import *
from tkinter import filedialog
from glob import glob
import operator
import formulas as fo


class Metadata:
    def __init__(
        self,
        *,
        pixel_size: float = 0.000024,
        framestep_size: float = 0.0217,
        frame_cutoff: int = 10,
    ) -> None:
        """Initialize microscope parameters

        Args:
            pixel_size (float, optional): pixel width. Defaults to 0.000024.
            framestep_size (float, optional): time between frames. Defaults to 0.0217.
            frame_cutoff (int, optional): minimum trajectory length in frames. Defaults to 10.
        """
        self.pixel_size = pixel_size
        self.framestep_size = framestep_size
        self.frame_cutoff = frame_cutoff


class Reader:
    def __init__(self):
        self.df = None
        root = Tk()
        root.withdraw()

    def combine_files(self):
        while True:
            rootdir = filedialog.askdirectory()
            filenames = glob(rootdir + "/*.csv")
            if filenames:
                self.df = pd.concat(map(pd.read_csv, filenames))
                break
            else:
                print("no csv files found in selected")
                exit = input("Continue? : (y/n)")
                if exit == "n":
                    sys.exit()

    def single_file(self):
        try:
            file_name = filedialog.askopenfilename()
            if file_name.endswith(".csv"):
                self.df = pd.read_csv(file_name, index_col=[0])
            elif file_name.endswith(".xlsx"):
                self.df = pd.read_excel(file_name, index_col=[0])
            else:
                raise TypeError
        except TypeError:
            print(f"{os.path.splitext(file_name)[1]} --- File type not supported")


class Analyze:
    def __init__(self, metadata: object, df: pd.DataFrame, brightness_col: str = "m2"):
        self._metadata = metadata
        self.df = df
        self._brightness_col = brightness_col

    def trio_lyze(self):
        """Calculate average brightness, length, and diffusivity for each trajectory

        Args:
            df (pd.DataFrame): trajectory data

        Returns:
            pd.DataFrame: updated trajectory data with trio
        """
        self.df["Average_Brightness"] = self.df.groupby("Trajectory")[
            self._brightness_col
        ].transform(np.mean)
        self.df["Length (frames)"] = self.df.groupby("Trajectory")[
            "Trajectory"
        ].transform("size")
        data = self.df.groupby("Trajectory")[["x", "y"]].apply(
            fo.Calc.one_step_MSD, self._metadata
        )
        data = pd.DataFrame(data.to_list(), columns=["SDs", "MSD"])
        data.index += 1
        data.reset_index(inplace=True)
        data = data.rename(columns={"index": "Trajectory"})
        self.df["SDs"] = reduce(operator.add, data["SDs"])
        self.df = self.df.merge(data[["Trajectory", "MSD"]])
        return self

    def identify(
        self,
        *,
        criteria: dict = {
            "dim": ("Average_Brightness", "<", 3.1),
            "bright": ("Average_Brightness", ">", 3.8),
            "short": ("Length (frames)", "<", 10),
            "slow": ("MSD", "<", 0.3),
            "fast": ("MSD", ">", 3.5),
        },
    ):
        ops = {
            "<": operator.lt,
            "=<": operator.le,
            ">": operator.gt,
            "=>": operator.ge,
            "=": operator.eq,
        }
        for key, val in criteria.items():
            col, op, num = val[0], val[1], val[2]
            self.df.loc[ops[op](self.df[col], num), key] = key
        self.df = self.df.fillna("")
        self.df["ID"] = self.df.iloc[:, -5:].sum(axis=1)
        return self


class Plot:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        x: str = "Average_Brightness",
        y: str = "Length (frames)",
        z: str = "MSD",
        id_col="ID",
    ):
        # assert {"Trajectory", "ID"} in set(df.columns)
        self.df = df[["Trajectory", x, y, z, "ID"]]
        self._x_label = x
        self._y_label = y
        self._z_label = z
        self._point_labels = id_col
        self._title = "Trajectories"

    def display(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel(self._x_label, labelpad=10)
        ax.set_ylabel(self._y_label, labelpad=10)
        ax.set_zlabel(self._z_label, labelpad=10)
        ax.set_title(self._title)
        groups = self.df.groupby(self.df[self._point_labels])
        for name, group in groups:
            ax.scatter3D(
                group[self._x_label],
                group[self._y_label],
                group[self._z_label],
                label=f"{group[self._point_labels].unique()[0]} : {len(group['Trajectory'].unique())}",
            )
        ax.view_init(30, 60)
        plt.legend(title="IDs", bbox_to_anchor=(2, 0.5), loc="right")
        # plt.legend(*sc.legend_elements(), title="Group ID")
        plt.show()


if __name__ == "__main__":
    meta = Metadata()
    file = Reader()
    file.single_file()
    data = Analyze(meta, file.df)
    data.trio_lyze().identify()
    figure = Plot(data.df)
    figure.display()
