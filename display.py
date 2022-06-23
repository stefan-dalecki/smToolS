"""Display figures"""

from math import ceil
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.ticker import PercentFormatter, MultipleLocator
from matplotlib import colors
import numpy as np
import pandas as pd
import formulas as fo


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
        self._data = data
        self._bins = ceil(max(data)) * 10
        self._x_label = x_label
        self._y_label = y_label
        self._title = title

    def plot(self) -> None:
        """Plot brightness data"""
        fig, axs = plt.subplots(tight_layout=True, figsize=(12, 5))
        N, bins, patches = axs.hist(
            self._data,
            bins=[i * 0.1 + 0.1 for i in range(self._bins)],
            edgecolor="black",
        )
        fracs = N / N.max()
        norm = colors.Normalize(fracs.min(), fracs.max())
        axs.set_xlabel(self._x_label)
        axs.xaxis.set_major_locator(MultipleLocator(0.5))
        axs.xaxis.set_minor_locator(MultipleLocator(0.1))
        axs.set_xlim(0, 7)
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

    def __init__(
        self, df: pd.DataFrame, suggested_clusters: dict, NUM_ClUSTERS: int
    ) -> None:
        self._df = df
        self._suggested_clusters = suggested_clusters
        self._NUM_CLUSTERS = NUM_ClUSTERS
        self._x = None
        self._x_label = None
        self._y = None
        self._y_label = None
        self._z = None
        self._z_label = None
        self._title = None

    def set_attributes(self) -> None:
        """set figure attributes"""
        self._x = self._df.iloc[:, 0]
        self._x_label = self._x.name
        self._y = self._df.iloc[:, 1]
        self._y_label = self._y.name
        self._z = self._df.iloc[:, 2]
        self._z_label = self._z.name
        self._title = f"K-Means Clustering (k = {self._NUM_CLUSTERS})\n \
            Suggested Clusters : {self._suggested_clusters}"
        return self

    def plot(self) -> None:
        """Plot the figure"""
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
            cmap="turbo",
        )
        ax.view_init(30, 60)
        plt.legend(*sc.legend_elements(), title="Group ID")
        plt.show()


# class MSDLine:
#     def __init__(self, df: pd.DataFrame) -> None:
#         self.data = df
#         self._x_label = None

#     def display(self):
#         pass


class ScatteredLine:
    """Line overlaying scatter data"""

    def __init__(self, model: object) -> None:
        """Initialize plot object

        Args:
            model (object): Model class object
        """
        self.model = model
        self._x_label = None
        self._y_label = None
        self._main_title = None

    def set_labels(self) -> None:
        """Establish plot axis labels"""
        self._x_label = self.model.kinetic.x_label
        self._y_label = self.model.kinetic.y_label
        self._main_title = self.model.movie.figure_title
        return self

    def plot(self, display: bool, save: bool) -> None:
        """Generate plot

        Args:
            display (bool): whether to display figure
            save (bool): whether to save figure
        """
        x_data = self.model.kinetic.table.iloc[:, 0].values.astype(float)
        y_data = self.model.kinetic.table.iloc[:, 1].values.astype(float)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5), tight_layout=True)
        fig.suptitle(f"{self.model.movie.figure_title} --- {self.model.kinetic.name}")
        fig.supxlabel(self._x_label)
        fig.supylabel(self._y_label)
        ax1.scatter(
            x_data,
            y_data,
            s=3,
            alpha=1,
            color="grey",
            label=f"Data: n= {fo.Calc.trajectory_count(self.model.movie.data_df)}",
        )
        ax1.set_title(self.model.model_name)
        ax1_label = ""
        if self.model.components > 1:
            ordinal = lambda n: "%d%s" % (
                n,
                "tsnrhtdd"[(n // 10 % 10 != 1) * (n % 10 < 4) * n % 10 :: 4],
            )
            populations = [
                "{0:.3f}".format(i / sum(self.model.popt[: self.model.components]))
                for i in self.model.popt[: self.model.components]
            ]
            population_covariances = [
                np.format_float_scientific(
                    i * self.model.metadata.framestep_size, precision=1, exp_digits=2
                )
                for i in self.model.pcov[: self.model.components]
            ]
            time_constants = [
                "{0:.3f}".format(i)
                for i in self.model.converted_popt[self.model.components :]
            ]
            time_constant_covariances = [
                np.format_float_scientific(i, precision=1, exp_digits=2)
                for i in self.model.converted_pcov[self.model.components :]
            ]

            for i in range(self.model.components):
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
                    + f" {self.model.kinetic.unit}\n"
                )

        elif self.model.components == 1:
            time_constant = "{0:.3f}".format(self.model.converted_popt[0])
            time_constant_covariance = np.format_float_scientific(
                self.model.converted_pcov[0], precision=1, exp_digits=2
            )
            ax1_label += (
                str(time_constant)
                + " \u00B1 "
                + str(time_constant_covariance)
                + f" {self.model.kinetic.unit}\n"
            )
        ax1_label.rstrip("\n")
        ax1_label += "\nR\u00b2 : {0:.3}".format(self.model.R2)
        ax1.plot(
            x_data,
            self.model.equation(x_data, *self.model.popt),
            color="white",
            lw=3,
            alpha=0.4,
        )
        ax1.plot(
            x_data,
            self.model.equation(x_data, *self.model.popt),
            label=ax1_label,
            color="blue",
            lw=1,
        )
        ax1.legend()
        ax2.plot(
            x_data,
            self.model.residuals,
            color="blue",
            lw=1,
        )
        ax2.hlines(y=0, xmin=0, xmax=max(x_data), color="grey", linestyle="--", lw=0.7)
        ax2.set_title("Residuals")
        if save:
            plt.savefig(
                f"{self.model.movie.filepath[:-4]}_{self.model.kinetic.name} \
                    _{self.model.model_name}".replace(
                    " ", ""
                )
            )
        if display:
            plt.show()
