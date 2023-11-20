"""
Filter trajectory data using brightness, length, and diffusion thresholding.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Optional, cast

# from kneed import KneeLocator
# from sklearn.cluster import KMeans
# from sklearn.metrics import silhouette_score
# from sklearn.preprocessing import StandardScaler
# from smToolS.analysis_tools import display as di
# from smToolS.analysis_tools import formulas as fo
# from smToolS.sm_helpers import constants as cons
from simo_tools import constants as cons
from simo_tools import metadata as meta

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


@dataclass
class Cutoff:
    """
    Length, Brightness, MSD, etc...
    """

    min: Optional[float] = None  # noqa: A003
    max: Optional[float] = None  # noqa: A003
    method: cons.CutoffMethods = cons.CutoffMethods.MANUAL
    display: bool = False
    save: bool = False

    def __post_init__(self):
        """
        The 'manual' method (default) requires 'high' and 'low cutoffs'.

        These are optional for the 'auto' method.

        """
        if self.method != cons.CutoffMethods.AUTO:
            assert self.min or self.max, (
                "If not using auto-thresholding, you must specify a 'ceil' or 'floor'"
                " cutoff."
            )

        self.min = self.min or 0.0
        self.max = self.max or float("inf")

    @property
    @abstractmethod
    def param(self):
        """
        One of cons.Cutoffs.
        """

    def threshold(self, trajs: meta.Trajectories) -> meta.Trajectories:
        """
        Return trajectories within high and low cutoffs.
        """

        method_map = {
            cons.CutoffMethods.AUTO: self.auto,
            cons.CutoffMethods.MANUAL: self.manual,
        }

        try:
            thresholded_trajs = method_map[self.method](trajs)
        except KeyError:
            raise TypeError(
                "Thresholding method must be one of:"
                f" {' ,'.join(cons.CutoffMethods.set_of_options())}."
            )

        if self.display:
            self._display(trajs, self.save)

        return thresholded_trajs

    def _manually_threshold(self, param: cons.Cutoffs, trajs: meta.Trajectories):
        """
        Eliminates trajectories outside of low/high cutoffs.
        """
        cuttoff_map = {
            cons.Cutoffs.BRIGHTNESS: "mean_brightness",
            cons.Cutoffs.LENGTH: "length",
            cons.Cutoffs.DIFFUSION: "mean_squared_displacement",
        }
        cutoff_param = cuttoff_map[param]

        min_val = cast(float, self.min)
        max_val = cast(float, self.max)
        valid_trajs = [
            traj for traj in trajs if min_val <= getattr(traj, cutoff_param) <= max_val
        ]
        return meta.Trajectories(valid_trajs)

    @abstractmethod
    def auto(self, trajs: meta.Trajectories) -> meta.Trajectories:
        """
        Threshold trajectories without being given high or low cutoffs.
        """

    @abstractmethod
    def manual(self, trajs: meta.Trajectories) -> meta.Trajectories:
        """
        Threshold trajectories using given high or low cutoffs.
        """

    @abstractmethod
    def _display(self, trajs: meta.Trajectories, save_figure: bool):
        """
        Display eliminated and kept trajectories.
        """


@dataclass
class Brightness(Cutoff):
    min_num: Optional[float] = None

    @property
    def param(self):
        return cons.Cutoffs.BRIGHTNESS

    def manual(self, trajs: meta.Trajectories) -> meta.Trajectories:
        return self._manually_threshold(cons.Cutoffs.BRIGHTNESS, trajs)

    def _display(self, trajs: meta.Trajectories):
        histogram = di.BrightnessHistogram(df[cons.AVERAGE_BRIGHTNESS].unique())
        histogram.plot()
        while True:
            low_out = float(input("Select the low brightness cutoff : "))
            high_out = float(input("Select the high brightness cutoff : "))
            rm_outliers_df = df[df[cons.AVERAGE_BRIGHTNESS].between(low_out, high_out)]
            rm_df = rm_outliers_df.reset_index(drop=True)
            print(f"Trajectories Remaining : {fo.trajectory_count(rm_df)}")
            move_on = input("Choose new cutoffs (0) or continue? (1) : ")
            if move_on == "1":
                break
        self.cutoff_df = rm_df


class Length(Cutoff):
    @property
    def param(self):
        return cons.Cutoffs.LENGTH

    def manual(self, trajs: meta.Trajectories) -> meta.Trajectories:
        return self._manually_threshold(cons.Cutoffs.LENGTH, trajs)


class Diffusion(Cutoff):
    @property
    def param(self):
        return cons.Cutoffs.DIFFUSION

    def manual(self, trajs: meta.Trajectories) -> meta.Trajectories:
        return self._manually_threshold(cons.Cutoffs.DIFFUSION, trajs)


# class Clustering:
#     """
#     K-means clustering thresholding.
#     """

#     CLUSTER = "cluster"
#     ELBOW = "elbow"
#     SILHOUETTE = "silhouette"

#     def __init__(self, df: pd.DataFrame) -> None:
#         """
#         Initialize clustering object.

#         Args:
#             df (pd.DataFrame): raw trajectory data

#         """
#         self._df = df
#         self.cluster_data = None
#         self._scaled_features = None
#         self._kmeans_kwargs = None
#         self.suggested_clusters = {self.ELBOW: None, self.SILHOUETTE: None}
#         self._n_clusters = None
#         self._cluster_of_interest = None
#         self.cutoff_df = None
#         self.min_length = None

#     def scale_features(self) -> Self:
#         self.cluster_data = (
#             self._df[
#                 [
#                     cons.AVERAGE_BRIGHTNESS,
#                     cons.LENGTH_W_UNITS,
#                     cons.MSD_W_UNITS,
#                 ]
#             ]
#             .drop_duplicates()
#             .reset_index(drop=True)
#         )
#         # Features must be scaled for proper k-means clustering
#         scaler = StandardScaler()
#         self._scaled_features = scaler.fit_transform(self.cluster_data)
#         return self

#     def estimate_clusters(self):
#         self._kmeans_kwargs = {
#             "init": "random",
#             "n_init": 10,
#             "max_iter": 300,
#             "random_state": 42,
#         }
#         sse = []
#         silhouette_scores = []
#         for i in range(1, 11):
#             kmeans = KMeans(n_clusters=i, **self._kmeans_kwargs)
#             kmeans.fit(self._scaled_features)
#             sse.append(kmeans.inertia_)
#             if i > 1:
#                 score = silhouette_score(self._scaled_features, kmeans.labels_)
#                 silhouette_scores.append([i, score])
#         kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
#         knee = kl.elbow
#         self.suggested_clusters[self.ELBOW] = knee
#         silhouette = silhouette_scores.index(max(silhouette_scores))
#         self.suggested_clusters[self.SILHOUETTE] = silhouette
#         print(f"Suggested Clusters : \n   {self.suggested_clusters}")
#         return self

#     def display(self) -> Self:
#         while True:
#             self._n_clusters = int(input("Model how many clusters? : "))
#             kmeans = KMeans(n_clusters=self._n_clusters, **self._kmeans_kwargs)
#             kmeans.fit(self._scaled_features)
#             temp_data = deepcopy(self.cluster_data)
#             temp_data[self.CLUSTER] = kmeans.labels_ + 1
#             di.ClusterPlot(
#                 temp_data, self.suggested_clusters, self._n_clusters
#             ).set_attributes().plot()
#             ans = fo.Form.input_bool("Display another model?")
#             if not ans:
#                 break
#         self._cluster_of_interest = int(
#             input("Which cluster would you like to keep? : ")
#         )
#         return self

#     def cluster(self) -> None:
#         kmeans = KMeans(n_clusters=self._n_clusters, **self._kmeans_kwargs)
#         kmeans.fit(self._scaled_features)
#         self.cluster_data[self.CLUSTER] = kmeans.labels_ + 1
#         keep_trajectories = self.cluster_data[
#             self.cluster_data[self.CLUSTER] == self._cluster_of_interest
#         ].index.values
#         self.cutoff_df = self._df.loc[self._df[cons.TRAJECTORY].isin(keep_trajectories)]
#         # A minimum trajectory length is necessary for bound lifetime modeling
#         self.min_length = np.min(
#             self.cutoff_df.groupby([cons.TRAJECTORY])[cons.TRAJECTORY].size()
#         )


# @dataclass
# class Cutoffs:
#     brightness: Brightness
#     length: Length
#     diffusion: Diffusion
#     clustering: Clustering

#     @classmethod
#     def from_kwargs(cls, kwargs_dict: dict[str, float])
# class Brightness:
#     """
#     Brightness thresholding.
#     """

#     def __init__(
#         self,
#         frame_cutoff: int,
#         df: pd.DataFrame,
#         method: Optional[str] = None,
#         cutoffs: Optional[Tuple[float, float]] = None,
#     ) -> None:
#         """
#         Initialize brightness object.

#         Args:
#             metadata: persistent metadata
#             df: trajectory data
#             method: cutoff approach. Defaults to None.
#             cutoffs: high and low brightness values, used for semi_automatic method

#         """
#         self._frame_cutoff = (
#             frame_cutoff  # only used for estimating remaining trajectories after cutoff
#         )
#         self._df = df
#         self._method = method
#         self._cutoffs = cutoffs
#         self.cutoff_df = None
#         self()

#     def __call__(self) -> None:
#         """
#         Calls chosen brightness thresholding method.
#         """
#         # Since the semi_auto method creates a dictionary as opposed to string
#         # it requires special unpacking when called
#         logger.info(f"Beginning ---{self._method}--- Brightness Cutoffs")
#         func = getattr(Brightness, self._method)
#         func(self)
#         logger.info(f"Completed ---{self._method}--- Brightness Cutoffs")

#     def manual(self) -> None:
#         """
#         Manually set min and max brightness values.
#         """
#         df = self._df
#         histogram = di.BrightnessHistogram(df[cons.AVERAGE_BRIGHTNESS].unique())
#         histogram.plot()
#         while True:
#             low_out = float(input("Select the low brightness cutoff : "))
#             high_out = float(input("Select the high brightness cutoff : "))
#             rm_outliers_df = df[df[cons.AVERAGE_BRIGHTNESS].between(low_out, high_out)]
#             rm_df = rm_outliers_df.reset_index(drop=True)
#             print(f"Trajectories Remaining : {fo.trajectory_count(rm_df)}")
#             move_on = input("Choose new cutoffs (0) or continue? (1) : ")
#             if move_on == "1":
#                 break
#         self.cutoff_df = rm_df

#     def semi_auto(self) -> None:
#         """
#         Semi-Auto thresholding.

#         Uses the same cutoffs, set prior to script runtime, to
#         threshold all movies

#         """
#         df = self._df
#         rm_outliers_df = df[df[cons.AVERAGE_BRIGHTNESS].between(*self._cutoffs)]
#         rm_df = rm_outliers_df.reset_index(drop=True)
#         self.cutoff_df = rm_df

#     def auto(self) -> None:
#         """
#         Automatically calculate brightness cutoffs.

#         Cutoffs are determined by separating all brightness values
#         into '100' bins. From there, bins are selected sequentially,
#         right and left, from the 'mode' brightness bin until the
#         amount of remaining trajectories is greater than '100'.

#         """
#         df = self._df
#         min_brightness = df[cons.AVERAGE_BRIGHTNESS].min()
#         max_brightness = df[cons.AVERAGE_BRIGHTNESS].max()
#         BINS = 100
#         step = (max_brightness - min_brightness) / BINS
#         bin_sdf = np.arange(min_brightness, max_brightness, step)
#         groups = 1
#         while groups <= BINS * 0.2:
#             single_traj = df.drop_duplicates(subset=cons.TRAJECTORY, keep="first")
#             sdf = (
#                 single_traj.groupby(
#                     pd.cut(single_traj[cons.AVERAGE_BRIGHTNESS], bins=bin_sdf)
#                 )
#                 .size()
#                 .nlargest(groups)
#             )
#             cutoff_list = np.array([i.right and i.right for i in sdf.index])
#             low_out, high_out = round(np.min(cutoff_list), 3), round(
#                 np.max(cutoff_list), 3
#             )
#             rm_outliers_df = df[df[cons.AVERAGE_BRIGHTNESS].between(low_out, high_out)]
#             rm_df = rm_outliers_df.reset_index(drop=True)
#             grp_df = (
#                 rm_df.groupby(cons.TRAJECTORY)
#                 .filter(lambda x: len(x) > self._frame_cutoff)
#                 .reset_index(drop=True)
#             )
#             # The minimum expectation for the number of trajectories can be changed
#             # based on the number necessary for latter calculations
#             if fo.trajectory_count(grp_df) < 150:
#                 groups += 1
#                 continue
#             else:
#                 break
#         self.cutoff_df = rm_df


# class Length:
#     """
#     Remove too short of trajectories.
#     """

#     def __init__(self, length: int, df: pd.DataFrame, *, method: str = None) -> None:
#         """
#         Initialize length object.

#         Args:
#             length (int): metadata class object
#             df (pd.DataFrame): trajectory data
#             method (str, optional): way to filter trajectories. Defaults to None.

#         """
#         self._length = length
#         self._df = df
#         self._method = method
#         self.cutoff_df = None
#         self()

#     def __call__(self) -> None:
#         """
#         Call length filtering method.
#         """
#         logger.info(f"Beginning ---{self._method}--- Length Cutoffs")
#         func = getattr(Length, self._method)
#         func(self)
#         logger.info(f"Completed ---{self._method}--- Length Cutoffs")

#     def minimum(self) -> None:
#         """
#         Only use trajectories that are at least -x- frames long.
#         """
#         self.cutoff_df = (
#             self._df.groupby(cons.TRAJECTORY)
#             .filter(lambda x: len(x) > self._length)
#             .reset_index(drop=True)
#         )


# class Diffusion:
#     """
#     Diffusion cutoffs.
#     """

#     def __init__(
#         self,
#         df: pd.DataFrame,
#         diffusion_cutoffs: Optional[Tuple[float, float]] = None,
#     ) -> None:
#         """
#         Initialize diffusion object.

#         Args:
#             diffusion_cutoffs (Optional[Tuple[float, float]]):
#                 low/high cutoff value (um^2/sec)
#             df (pd.DataFrame): trajectory data

#         """
#         self._low, self._high = diffusion_cutoffs or (0, float("inf"))
#         self._df = df
#         self.cutoff_df = None
#         self()

#     def __call__(self) -> None:
#         """
#         Call displacement cutoff function.
#         """
#         logger.info("Beginning Diffusion Cutoffs.")
#         self.displacement()
#         logger.info("Completed Diffusion Cutoffs.")

#     def displacement(self) -> None:
#         """
#         Use mean square displacement to filter trajectories.
#         """
#         df = self._df
#         rm_outliers_df = df[df[cons.MSD_W_UNITS].between(self._low, self._high)]
#         rm_df = rm_outliers_df.reset_index(drop=True)
#         self.cutoff_df = rm_df
