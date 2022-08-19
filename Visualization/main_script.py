import os
import sys
import operator
from tkinter import filedialog
from math import ceil
from functools import reduce
from glob import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib import colors
import formulas as fo


class Metadata:
    """Microscope metadata"""

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

    def combine_files(self):
        """Read and combine multiple files"""
        # This can only be done when files are already pre_processed by smToolS
        print("Select your folder...")
        while True:
            rootdir = filedialog.askdirectory()
            print("Grouping Files...")
            filenames = glob(rootdir + "/*pre_processed.csv")
            if filenames:
                self.df = pd.concat(map(pd.read_csv, filenames))
                break
            else:
                print("no csv files found in selected directory")
                sys.exit()
        print("Files grouped successfully")

    def single_file(self):
        print("Select your file")
        try:
            file_name = filedialog.askopenfilename()
            print(f"{file_name}\nLoading file...")
            if file_name.endswith(".csv"):
                self.df = pd.read_csv(file_name, index_col=[0])
            elif file_name.endswith(".xlsx"):
                self.df = pd.read_excel(file_name)
            else:
                raise TypeError
        except TypeError:
            print(f"{os.path.splitext(file_name)[1]} --- File type not supported")
        print("File loaded successfully")


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
        data = pd.DataFrame(data.to_list(), columns=["SDs", "MSD (\u03BCm\u00b2/sec)"])
        data.index += 1
        data.reset_index(inplace=True)
        data = data.rename(columns={"index": "Trajectory"})
        self.df["SDs"] = reduce(operator.add, data["SDs"])
        self.df = self.df.merge(data[["Trajectory", "MSD (\u03BCm\u00b2/sec)"]])
        return self

    def identify(
        self,
        sep: str = "\u00B7",
        keepers: str = "Valid",
        *,
        criteria: dict = {
            "dim": ("Average_Brightness", "<", 3.1),
            "bright": ("Average_Brightness", ">", 3.8),
            "short": ("Length (frames)", "<", 10),
            "slow": ("MSD (\u03BCm\u00b2/sec)", "<", 0.3),
            "fast": ("MSD (\u03BCm\u00b2/sec)", ">", 3.5),
        },
    ):
        """Draw descriptions for values in specified dataframe column

        Args:
            sep (str, optional): what will separate multiple descriptions (i.e. dim '+' short).
                Defaults to "\u00B7".
            keepers (str, optional): string for trajectories that pass cutoffs.
                Defaults to "Valid".
            criteria (dict, optional): Descriptor key with tuple value containing
                column name, operator, and float criterion.
                Defaults to { "dim": ("Average_Brightness", "<", 3.1), "bright": ("Average_Brightness", ">", 3.8),
                "short": ("Length (frames)", "<", 10), "slow": ("MSD (\u03BCm\u00b2/sec)", "<", 0.3), "fast": ("MSD (\u03BCm\u00b2/sec)", ">", 3.5), }.

        """
        ops = {
            "<": operator.lt,
            "=<": operator.le,
            ">": operator.gt,
            "=>": operator.ge,
            "=": operator.eq,
        }
        for key, val in criteria.items():
            col, op, num = val[0], val[1], val[2]
            self.df.loc[ops[op](self.df[col], num), key] = f"{key} {sep} "
        self.df = self.df.fillna("")
        self.df["ID"] = self.df.iloc[:, -len(criteria) :].sum(axis=1)
        # Removes ending from last addition
        self.df["ID"] = self.df["ID"].str.rstrip(f" {sep} ")
        # Trajectories with no tag are set to the keepers variable
        self.df.loc[self.df["ID"] == "", "ID"] = keepers
        return self


class Plot:
    def __init__(
        self,
        df: pd.DataFrame,
        *,
        x: str = "Average_Brightness",
        y: str = "Length (frames)",
        z: str = "MSD (\u03BCm\u00b2/sec)",
        id_col="ID",
    ):
        assert {"Trajectory", x, y, z, id_col} and set(df.columns)
        self.df = df[["Trajectory", x, y, z, "ID"]]
        self._unique_df = self.df.drop_duplicates(subset="Trajectory")
        self._x_label = x
        self._y_label = y
        self._z_label = z
        self._point_labels = id_col
        self._title = "Trajectory Characteristics"

    def TwoD_Histograms(self):
        for i in [self._x_label, self._y_label, self._z_label]:
            data = self._unique_df[i]
            bins = ceil(max(data)) * 10
            fig, axs = plt.subplots(tight_layout=True, figsize=(12, 5))
            # Tick marks are set relative to the scale on the x-axis
            sep = 10 ** (int(f"{bins:.3e}".split("+")[-1]) - 2)
            N, bins, patches = axs.hist(
                data,
                bins=[i * sep + sep for i in range(bins)],
                edgecolor="black",
            )
            fracs = N / N.max()
            norm = colors.Normalize(fracs.min(), fracs.max())
            axs.set_xlabel(i)
            axs.xaxis.set_major_locator(MultipleLocator(sep * 5))
            axs.xaxis.set_minor_locator(MultipleLocator(sep))
            # No negative brightness values should exist
            axs.set_xlim(0, ceil(max(data)))
            axs.set_ylabel("Frequency (#)")
            axs.set_title(f"Binned {i}")
            for thisfrac, thispatch in zip(fracs, patches):
                color = plt.cm.plasma(norm(thisfrac))
                thispatch.set_facecolor(color)
            ax2 = axs.twinx()
            ax2.yaxis.set_major_formatter(PercentFormatter(xmax=len(data) / 100))
            plt.show(block=True)

    def ThreeD_Scatter(self):
        fig = plt.figure(figsize=(10, 5))
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
        plt.legend(title="IDs", bbox_to_anchor=(1.75, 0.5), loc="right")
        plt.show()


if __name__ == "__main__":
    meta = Metadata()
    file = Reader()
    file.single_file()
    image_data = Analyze(meta, file.df)
    if "Average_Brightness" in image_data.df:
        image_data.identify()
    else:
        image_data.trio_lyze().identify()
    figure = Plot(image_data.df)
    figure.TwoD_Histograms()
    figure.ThreeD_Scatter()
