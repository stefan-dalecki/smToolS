"""Display figures"""
import logging
import os
from math import ceil
from typing import Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, PercentFormatter

from smToolS.analysis_tools import curvefitting
from smToolS.analysis_tools import formulas as fo
from smToolS.sm_helpers import constants as cons

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BrightnessHistogram:
    """Histogram of Brightness values"""

    def __init__(
        self,
        data: list,
        *,
        x_label: str = "Intensity / Brightness",
        y_label: str = "Frequency",
        title: str = "Binned Brightness",
    ) -> None:
        if min(data) < 0:
            raise ValueError(f"{min(data)} : Brightness value cannot be negative.")
        self._data = data
        # Bins are set based on a rounded version of the max value
        self._bins = ceil(max(data)) * 10
        self._x_label = x_label
        self._y_label = y_label
        self._title = title

    def plot(self) -> None:
        """Plot brightness data"""
        fig, axs = plt.subplots(tight_layout=True, figsize=(12, 5))
        # Tick marks are set relative to the scale on the x-axis
        sep = 10 ** (int(f"{self._bins:.3e}".split("+")[-1]) - 2)
        N, bins, patches = axs.hist(
            self._data,
            bins=[i * sep + sep for i in range(self._bins)],
            edgecolor="black",
        )
        fracs = N / N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        axs.set_xlabel(self._x_label)
        axs.xaxis.set_major_locator(MultipleLocator(sep * 5))
        axs.xaxis.set_minor_locator(MultipleLocator(sep))
        # No negative brightness values should exist
        axs.set_xlim(0, ceil(max(self._data)))
        axs.set_ylabel(self._y_label)
        axs.set_title(self._title)
        for thisfrac, thispatch in zip(fracs, patches):
            color = plt.cm.plasma(norm(thisfrac))
            thispatch.set_facecolor(color)
        ax2 = axs.twinx()
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=len(self._data) / 100))
        plt.show(block=False)


class ThreeDScatter:
    """Three dimensional scatter plot"""

    def __init__(self, df: pd.DataFrame, suggested_clusters: dict, num_clusters: int) -> None:
        self._df = df
        self._suggested_clusters = suggested_clusters
        self._num_clusters = num_clusters
        self._x = None
        self._x_label = None
        self._y = None
        self._y_label = None
        self._z = None
        self._z_label = None
        self._title = None

    def set_attributes(self) -> Self:
        """set figure attributes"""
        # First three columns must contain desired data
        self._x = self._df.iloc[:, 0]
        self._x_label = self._x.name
        self._y = self._df.iloc[:, 1]
        self._y_label = self._y.name
        self._z = self._df.iloc[:, 2]
        self._z_label = self._z.name
        self._title = f"K-Means Clustering (k = {self._num_clusters})\n \
            Suggested Clusters : {self._suggested_clusters}"
        return self

    def plot(self) -> None:
        """Plot the figure"""
        plt.close()
        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(projection="3d")
        # x_min = int(np.min(self._x.values))
        # x_max = int(np.max(self._x.values))
        # y_min = int(np.min(self._y.values))
        # y_max = int(np.max(self._y.values))
        # z_min = int(np.min(self._z.values))
        # z_max = int(np.max(self._z.values))
        # xx, yy = np.meshgrid(
        #     range(x_min, x_max),
        #     range(y_min, y_max),
        # )
        # cluster_id = 1
        # z_minlength = np.min(self._df[self._df["Cluster"] == cluster_id]["Length"])
        # z_maxlength = np.max(self._df[self._df["Cluster"] == cluster_id]["Length"])
        # small_Z = z_minlength - xx
        # large_Z = z_maxlength - xx
        # ax.plot_surface(xx, small_Z, yy, alpha=0.5, color="lightblue")
        # ax.plot_surface(xx, large_Z, yy, alpha=0.5, color="lightblue")
        # for t in ax.zaxis.get_major_ticks():
        #     t.label.set_fontsize(10)
        ax.set_xlabel(self._x_label, labelpad=10)
        ax.set_ylabel(self._y_label, labelpad=10)
        ax.set_zlabel(self._z_label, labelpad=10)
        # ax.set_xlim([x_min, x_max])
        # ax.set_ylim([y_min, y_max])
        # ax.set_zlim([z_min, z_max])
        ax.set_title(self._title)
        sc = ax.scatter3D(
            self._x,
            self._y,
            self._z,
            c=self._df["Cluster"],
            cmap="Dark2",
        )
        ax.view_init(30, 60)
        plt.legend(*sc.legend_elements(), title="Group ID")
        plt.show(block=False)


# class MSDLine:
#     def __init__(self, kinetic: object, df: pd.DataFrame, line: object) -> None:
#         self._kinetic = kinetic
#         self._df = df
#         self._line = line
#         self._x = None
#         self._x_label = None
#         self._y = None
#         self._y_label = None

#     def set_attributes(self) -> None:
#         """set figure attributes"""
#         self._x = self._df.iloc[:, 0]
#         self._x_label = self._x.name
#         self._y = self._df.iloc[:, 1]
#         self._y_label = self._y.name

#     def plot(self):
#         x_data = self._x.values.astype(float)
#         y_data = self._y.values.astype(float)
#         fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
#         fig.suptitle(f"{self._kinetic.name} ({self._kinetic.unit})")
#         fig.supxlabel(self._x_label)
#         fig.supylabe(self._y_label)
#         ax1.scatter(x_data, y_data, s=3, alpha=1, color="grey", label="Data")
#         ax1.set_title("Linear Regression")
#         ax1.plot(
#             x_data,
#         )


class ScatteredLine:
    """Line overlaying scatter data"""

    def __init__(self, model: curvefitting.Model) -> None:
        """Initialize plot object

        Args:
            model (object): Model class object
        """
        self._model = model

    @property
    def _x_label(self):
        return self._model.kinetic.x_label

    @property
    def _y_label(self):
        return self._model.kinetic.y_label

    @property
    def _main_title(self):
        return self._model.movie.figure_title

    def plot(self, display: bool, save: bool, save_location: str) -> None:
        """Generate plot

        Args:
            display (bool): whether to display figure
            save (bool): whether to save figure
        """
        logger.info(f"Building ---{__class__.__name__}--- for '{self._model.model_name}'.")
        x_data = self._model.kinetic.table.iloc[:, 0].values.astype(float)
        y_data = self._model.kinetic.table.iloc[:, 1].values.astype(float)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
        fig.suptitle(f"{self._model.movie.figure_title} --- {self._model.kinetic.name}")
        fig.supxlabel(self._x_label)
        fig.supylabel(self._y_label)
        ax1.scatter(
            x_data,
            y_data,
            s=3,
            alpha=1,
            color="grey",
            label=f"Data: n= {fo.trajectory_count(self._model.movie.data_df)}",
        )
        ax1.set_title(self._model.model_name)
        ax1_label = ""
        # Model components follow a standardized format that can be broken down
        # by using the lines below
        if self._model.components > 1:
            ordinal = lambda n: "%d%s" % (
                n,
                "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
            )
            populations = [
                "{0:.3f}".format(i / sum(self._model.popt[: self._model.components]))
                for i in self._model.popt[: self._model.components]
            ]
            population_covariances = [
                np.format_float_scientific(
                    i * self._model.microscope.framestep_size, precision=1, exp_digits=2
                )
                for i in self._model.pcov[: self._model.components]
            ]
            time_constants = [
                "{0:.3f}".format(i) for i in self._model.converted_popt[self._model.components :]
            ]
            time_constant_covariances = [
                np.format_float_scientific(i, precision=1, exp_digits=2)
                for i in self._model.converted_pcov[self._model.components :]
            ]

            for i in range(self._model.components):
                ax1_label += (
                    ordinal(i + 1)
                    + " Frac: "
                    + str(populations[i])
                    + " \u00B1 "
                    + str(population_covariances[i])
                )
                ax1_label += (
                    f"\n   {time_constants[i]}"
                    + " \u00B1 "
                    + str(time_constant_covariances[i])
                    + f" {self._model.kinetic.unit}\n"
                )
        # There is no 'else' catch since the type of models is meant to be scallable
        elif self._model.components == 1:
            time_constant = "{0:.3f}".format(self._model.converted_popt[0])
            time_constant_covariance = np.format_float_scientific(
                self._model.converted_pcov[0], precision=1, exp_digits=2
            )
            ax1_label += (
                str(time_constant)
                + " \u00B1 "
                + str(time_constant_covariance)
                + f" {self._model.kinetic.unit}\n"
            )
        ax1_label.rstrip("\n")
        ax1_label += "\nR\u00b2 : {0:.3}".format(self._model.R2)
        ax1.plot(
            x_data,
            self._model.equation(x_data, *self._model.popt),
            color="white",
            lw=3,
            alpha=0.4,
        )
        ax1.plot(
            x_data,
            self._model.equation(x_data, *self._model.popt),
            label=ax1_label,
            color="blue",
            lw=1,
        )
        ax1.legend()
        ax2.plot(
            x_data,
            self._model.residuals,
            color="blue",
            lw=1,
        )
        ax2.hlines(y=0, xmin=0, xmax=max(x_data), color="grey", linestyle="--", lw=0.7)
        ax2.set_title("Residuals")
        if save:
            filename = os.path.join(
                save_location,
                f"{self._model.movie.name[cons.FILENAME]}_{self._model.kinetic.name}_{self._model.model_name}".replace(
                    " ", ""
                ),
            )
            logger.info(f"Saving ---{__class__.__name__}--- to '{filename}'.")
            plt.savefig(filename)
        if display:
            logger.info(f"Displaying ---{__class__.__name__}--- for '{self._model.model_name}'.")
            plt.show()
        plt.close()


class VisualizationPlots:
    _id_col = "ID"

    def __init__(
        self,
        df: pd.DataFrame,
        display: bool,
        save_figures: bool,
        save_location: str,
        save_name: str,
        *,
        _x: str = cons.AVERAGE_BRIGHTNESS,
        _y: str = cons.LENGTH_W_UNITS,
        _z: str = cons.MSD_W_UNITS,
    ):
        self.df = df[[cons.TRAJECTORY, _x, _y, _z, self._id_col]]
        self._display = display
        self._save_figures = save_figures
        self._save_path = os.path.join(save_location, save_name)
        self._unique_df = self.df.drop_duplicates(subset=cons.TRAJECTORY)
        self._x_label = _x
        self._y_label = _y
        self._z_label = _z
        self._title = "Trajectory Characteristics"

    def two_dimensional_histogram(self):
        FUNC_NAME = "two_dimensional_histogram"
        for i in [self._x_label, self._y_label, self._z_label]:
            logger.info(f"Building '{i}' ---{FUNC_NAME}--.")
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
            if self._save_figures:
                full_path = f"{self._save_path}_{FUNC_NAME}"
                logger.info(f"Saving ---{FUNC_NAME}--- to '{full_path}'.")
                plt.savefig(full_path)
            if self._display:
                logger.info(f"Displaying ---{FUNC_NAME}---.")
                plt.show(block=True)

    def three_dimensional_scatter(self):
        FUNC_NAME = "three_dimensional_scatter"
        logger.info(f"Building ---{FUNC_NAME}---.")
        fig = plt.figure(figsize=(10, 5))
        ax = fig.add_subplot(projection="3d")
        ax.set_xlabel(self._x_label, labelpad=10)
        ax.set_ylabel(self._y_label, labelpad=10)
        ax.set_zlabel(self._z_label, labelpad=10)
        ax.set_title(self._title)
        groups = self.df.groupby(self.df[self._id_col])
        for name, group in groups:
            ax.scatter3D(
                group[self._x_label],
                group[self._y_label],
                group[self._z_label],
                label=f"{group[self._id_col].unique()[0]} : {len(group[cons.TRAJECTORY].unique())}",
            )
        ax.view_init(30, 60)
        plt.legend(title=f"{self._id_col}s", bbox_to_anchor=(1.75, 0.5), loc="right")
        if self._save_figures:
            full_path = f"{self._save_path}_{FUNC_NAME}"
            logger.info(f"Saving ---{FUNC_NAME}--- to '{full_path}'.")
            plt.savefig(full_path)
        if self._display:
            logger.info(f"Displaying ---{FUNC_NAME}---.")
            plt.show()
