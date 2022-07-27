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
    def __init__(self, df: pd.DataFrame):
        self._df = df

    def trio_lyze(self):
        """Calculate average brightness, length, and diffusivity for each trajectory

        Args:
            df (pd.DataFrame): trajectory data

        Returns:
            pd.DataFrame: updated trajectory data with trio
        """
        self._df["Average_Brightness"] = self._df.groupby("Trajectory")[
            "Brightness"
        ].transform(np.mean)
        self._df["Length (frames)"] = self._df.groupby("Trajectory")[
            "Trajectory"
        ].transform("size")
        data = self._df.groupby("Trajectory")[["x", "y"]].apply(fo.Calc.one_step_MSD)
        data = pd.DataFrame(data.to_list(), columns=["SDs", "MSD"])
        data.index += 1
        data.reset_index(inplace=True)
        data = data.rename(columns={"index": "Trajectory"})
        self._df["SDs"] = reduce(operator.add, data["SDs"])
        self._df = self._df.merge(data[["Trajectory", "MSD"]])
        self._df = self._df.assign(Group="")
        return self

    def identify(
        self,
        *,
        brightness: dict = {"dim": 3.1, "bright": 3.8},
        min_length: dict = {"short": 10},
        diffusion: dict = {"slow": 0.3, "fast": 3.5},
    ):
        assert min_length["short"] < np.max(self._df["Length (frames)"])
        
    def identifying_function()
