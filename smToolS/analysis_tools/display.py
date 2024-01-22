"""Display figures."""

import abc
import logging
import os
from abc import abstractmethod
from math import ceil
from typing import Dict, List, Optional, Self

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import colors
from matplotlib.ticker import MultipleLocator, PercentFormatter

from smToolS.analysis_tools import curvefitting
from smToolS.analysis_tools import formulas as fo
from smToolS.analysis_tools import kinetics as kin
from smToolS.sm_helpers import constants as cons

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class BasePlotter(metaclass=abc.ABCMeta):
    """Abstract base plot class that supports up to three-dimensional
    figures."""

    def __init__(self):
        pass

    @property
    @abstractmethod
    def _x_data(self):
        """x-axis values."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _x_label(self):
        """x-axis label."""
        return NotImplementedError

    @property
    @abstractmethod
    def _y_data(self):
        """y-axis values."""
        raise NotImplementedError

    @property
    @abstractmethod
    def _y_label(self):
        """y-axis label."""
        return NotImplementedError

    @property
    @abstractmethod
    def _z_data(self):
        """z-axis data.

        Will raise error if used on two-dimensional plots
        """
        raise NotImplementedError

    @property
    @abstractmethod
    def _z_label(self):
        """z-label

        Will raise error if used on two-dimensional plots."""
        return NotImplementedError

    @property
    @abstractmethod
    def _title(self):
        """figure title."""
        return NotImplementedError

    @abstractmethod
    def plot(self):
        """All classes must have a method to plot their data."""
        raise NotImplementedError


class BrightnessHistogram:
    """Histogram of Brightness values."""

    # default axis and title values
    _X_LABEL = "Intensity / Brightness"
    _Y_LABEL = "Frequency"
    _TITLE = "Binned Brightness"

    def __init__(
        self,
        data: List[float],
        *,
        x_label: Optional[str] = None,
        y_label: Optional[str] = None,
        title: Optional[str] = None,
    ) -> None:
        """Generates a brightness histogram from a list of brightness values
        (floats)

        Args:
            data: list of brightness values
            x_label: optional override for x-axis label
            y_label: optional override for y-axis label
            title: optional override for figure title
        """
        if min(data) < 0:
            raise ValueError(f"{min(data)} : Brightness value cannot be negative.")
        self._data = data
        # Bins are set based on a rounded version of the max value
        self._bins = ceil(max(data)) * 10
        self._x_label = x_label or self._X_LABEL
        self._y_label = y_label or self._Y_LABEL
        self._title = title or self._TITLE

    def plot(self) -> None:
        """Plot brightness data."""
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

        # Larger fractions have more intense color
        for frac, patch in zip(fracs, patches):
            color = plt.cm.plasma(norm(frac))
            patch.set_facecolor(color)
        ax2 = axs.twinx()
        ax2.yaxis.set_major_formatter(PercentFormatter(xmax=len(self._data) / 100))
        plt.show(block=False)


class ClusterPlot:
    """Three-dimensional scatter plot."""

    def __init__(
        self, df: pd.DataFrame, suggested_clusters: Dict[str, int], num_clusters: int
    ) -> None:
        """This plot is used to help determine the amount of clusters to use
        for brightness thresholding.

        The implementation of cutoffs is handled outside of this class. Here, we strictly want to
        determine how many groups of trajectories we have in our movie. In some cases,
        biology should take precedence over these calculations. For example, if two proteins are
        imaged that cannot form dimers, you might only have 2-3 clusters, with an extra cluster
        perhaps representing photo-bleaching.

        Args:
            df: trajectory data
            suggested_clusters: elbow and silhouette score clustering suggestions
            num_clusters: actual number of clusters displayed
        """

        self._df = df
        self._suggested_clusters = suggested_clusters
        self._num_clusters = num_clusters

        # set using the function `set_attributes()`
        self._x = None
        self._x_label = None
        self._y = None
        self._y_label = None
        self._z = None
        self._z_label = None
        self._title = None

    def set_attributes(self) -> Self:
        """set figure attributes."""
        # First three columns must contain desired data
        self._x = self._df.iloc[:, 0]
        self._x_label = self._x.name  # axis label is column name
        self._y = self._df.iloc[:, 1]
        self._y_label = self._y.name  # axis label is column name
        self._z = self._df.iloc[:, 2]
        self._z_label = self._z.name  # axis label is column name
        self._title = (
            f"K-Means Clustering (k = {self._num_clusters})\nSuggested Clusters :"
            f" {self._suggested_clusters}"
        )
        return self  # allows for chained method calls

    def plot(self) -> None:
        """Plot the figure."""
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


class MSDLine:
    def __init__(self, kinetic: kin.Kinetic, df: pd.DataFrame, line: None) -> None:
        self._kinetic = kinetic
        self._df = df
        self._line = line
        self._x = None
        self._x_label = None
        self._y = None
        self._y_label = None

    def set_attributes(self) -> None:
        """set figure attributes."""
        self._x = self._df.iloc[:, 0]
        self._x_label = self._x.name
        self._y = self._df.iloc[:, 1]
        self._y_label = self._y.name

    def plot(self):
        x_data = self._x.values.astype(float)
        y_data = self._y.values.astype(float)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
        fig.suptitle(f"{self._kinetic.name} ({self._kinetic.unit})")
        fig.supxlabel(self._x_label)
        fig.supylabe(self._y_label)
        ax1.scatter(x_data, y_data, s=3, alpha=1, color="grey", label="Data")
        ax1.set_title("Linear Regression")
        ax1.plot(
            x_data,
        )


class ScatteredLine:
    """Line overlaying scatter data."""

    def __init__(self, model: curvefitting.Model) -> None:
        """Initialize plot object.

        Args:
            model: Model class object used for displaying figure
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
        """Generate plot.

        Args:
            display: whether to display figure
            save: whether to save figure
            save_location: filepath to save figure
        """
        logger.info(
            f"Building ---{__class__.__name__}--- for '{self._model.model_name}'."
        )
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

            def ordinal(n):
                return "%d%s" % (
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
                "{0:.3f}".format(i)
                for i in self._model.converted_popt[self._model.components :]
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
                    + " \u00b1 "
                    + str(population_covariances[i])
                )
                ax1_label += (
                    f"\n   {time_constants[i]}"
                    + " \u00b1 "
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
                + " \u00b1 "
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
                f"{self._model.movie.name[cons.FILENAME]}_{self._model.kinetic.name}_{self._model.model_name}"
                .replace(" ", ""),
            )
            logger.info(f"Saving ---{__class__.__name__}--- to '{filename}'.")
            plt.savefig(filename)
        if display:
            logger.info(
                f"Displaying ---{__class__.__name__}--- for '{self._model.model_name}'."
            )
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
                appendix = i.split(" ")[0]
                full_path = f"{self._save_path}_{FUNC_NAME}_{appendix}"
                logger.info(f"Saving ---{FUNC_NAME}--- to '{full_path}'.")
                self.df.to_csv(f"{full_path}_raw_data.csv")
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
        for _name, group in groups:
            ax.scatter3D(
                group[self._x_label],
                group[self._y_label],
                group[self._z_label],
                label=(
                    f"{group[self._id_col].unique()[0]} :"
                    f" {len(group[cons.TRAJECTORY].unique())}"
                ),
            )
        ax.view_init(30, 60)
        plt.legend(title=f"{self._id_col}s", bbox_to_anchor=(1.75, 0.5), loc="right")
        if self._save_figures:
            full_path = f"{self._save_path}_{FUNC_NAME}"
            logger.info(f"Saving ---{FUNC_NAME}--- to '{full_path}'.")
            self.df.to_csv(f"{full_path}_raw_data.csv")
            plt.savefig(full_path)
        if self._display:
            logger.info(f"Displaying ---{FUNC_NAME}---.")
            plt.show()
