import os
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
        root = Tk()
        root.withdraw()
        self._rootdir = filedialog.askdirectory()
        self._combined_df = None
        self._single_df = None

    def combine_files(self):
        filenames = glob(self._rootdir + "/*.csv")
        self._combined_df = pd.concat(map(pd.read_csv, filenames))

    def single_file(self, filename):
        self._single_df = pd.read_csv(os.path.join(self._rootdir, filename))


class Analyze:
    def __init__(self, metadata: object, df: pd.DataFrame):
        self._metadata = metadata
        self.df = df

    def trio_lyze(self):
        """Calculate average brightness, length, and diffusivity for each trajectory

        Args:
            df (pd.DataFrame): trajectory data

        Returns:
            pd.DataFrame: updated trajectory data with trio
        """
        self.df["Average_Brightness"] = self.df.groupby("Trajectory")[
            "Brightness"
        ].transform(np.mean)
        self.df["Length (frames)"] = self.df.groupby("Trajectory")[
            "Trajectory"
        ].transform("size")
        data = self.df.groupby("Trajectory")[["x", "y"]].apply(fo.Calc.one_step_MSD)
        data = pd.DataFrame(data.to_list(), columns=["SDs", "MSD"])
        data.index += 1
        data.reset_index(inplace=True)
        data = data.rename(columns={"index": "Trajectory"})
        self.df["SDs"] = reduce(operator.add, data["SDs"])
        self.df = self.df.merge(data[["Trajectory", "MSD"]])
        self.df = self.df.assign(Group="")
        return self

    def identify(
        self,
        *,
        brightness: dict = {"dim": 3.1, "bright": 3.8},
        min_length: dict = {"short": 10},
        diffusion: dict = {"slow": 0.3, "fast": 3.5},
    ):
        assert min_length["short"] < np.max(self.df["Length (frames)"])
        for row in self.df.iterrows:
            if row["Average_Brightness"] < brightness["dim"]:
                row["ID"] += "dim + "
            if row["Average_Brightness"] > brightness["bright"]:
                row["ID"] += "bright + "
            if row["Length (frames)"] < min_length["short"]:
                row["ID"] += "short + "
            if row["MSD"] < diffusion["slow"]:
                row["ID"] += "slow + "
            if row["MSD"] > diffusion["fast"]:
                row["ID"] += "fast + "
        self.df["ID"] = self.df["ID"].removesuffix(" + ").fillna("valid trajectory")
        return self


class Plot:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        x: str = "Average_Brightness",
        y: str = "Length (frames)",
        z: str = "MSD",
    ):
        self.df = df[["Trajectory", x, y, z, "ID"]]
        self._x = df[x]
        self._y = df[y]
        self._z = df[z]
        self._x_label = x
        self._y_label = y
        self._z_label = y
        self._title = "Trajectories"

    def figure(self):
        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.set_x_label(self._x_label, labelpad=10)
        ax.set_y_label(self._y_label, labelpad=10)
        ax.set_z_label(self._z_label, labelpad=10)
        ax.set_title(self._title)
        sc = ax.scatter3D(
            self._x,
            self._y,
            self._z,
            c=self.df["ID"],
            cmap="Dark2",
        )
        ax.view_init(30, 60)
        plt.legend(*sc.legend_elements(), title="Group ID")
        plt.show()


meta = Metadata()
file = Reader()
data = Analyze(meta, file)
figure = Plot(data.df)
