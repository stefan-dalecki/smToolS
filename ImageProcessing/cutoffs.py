"""Filter trajectory data"""

from copy import deepcopy
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import display as di
import formulas as fo
import decorators as dec


class Clustering:
    """K-means clustering thresholding"""

    class Clustering:
        """K-means clustering thresholding"""

    def __init__(self, metadata: object, df: pd.DataFrame) -> None:

        """Initialize clustering object

        Args:
            df (pd.DataFrame): raw trajectory data
        """
        self.metadata = metadata
        self._df = df
        self.cluster_data = None
        self._scaled_features = None
        self._kmeans_kwargs = None
        self.suggested_clusters = {"Elbow": None, "Silhouette": None}
        self._n_clusters = None
        self._cluster_of_interest = None
        self.cutoff_df = None
        self.min_length = None

    def scale_features(self) -> None:
        self.cluster_data = (
            self._df[
                [
                    "Average_Brightness",
                    "Length (frames)",
                    "MSD",
                ]
            ]
            .drop_duplicates()
            .reset_index(drop=True)
        )
        # Features must be scaled for proper k-means clustering
        scaler = StandardScaler()
        self._scaled_features = scaler.fit_transform(self.cluster_data)
        return self

    def estimate_clusters(self):
        self._kmeans_kwargs = {
            "init": "random",
            "n_init": 10,
            "max_iter": 300,
            "random_state": 42,
        }
        sse = []
        silhouette_scores = []
        for i in range(1, 11):
            kmeans = KMeans(n_clusters=i, **self._kmeans_kwargs)
            kmeans.fit(self._scaled_features)
            sse.append(kmeans.inertia_)
            if i > 1:
                score = silhouette_score(self._scaled_features, kmeans.labels_)
                silhouette_scores.append([i, score])
        kl = KneeLocator(range(1, 11), sse, curve="convex", direction="decreasing")
        knee = kl.elbow
        self.suggested_clusters["Elbow"] = knee
        silhouette = silhouette_scores.index(max(silhouette_scores))
        self.suggested_clusters["Silhouette"] = silhouette
        print(f"Suggested Clusters : \n   {self.suggested_clusters}")
        return self

    def display(self):
        while True:
            self._n_clusters = int(input("   Model how many clusters? : "))
            kmeans = KMeans(n_clusters=self._n_clusters, **self._kmeans_kwargs)
            kmeans.fit(self._scaled_features)
            temp_data = deepcopy(self.cluster_data)
            temp_data["Cluster"] = kmeans.labels_ + 1
            di.ThreeDScatter(
                temp_data, self.suggested_clusters, self._n_clusters
            ).set_attributes().plot()
            ans = fo.Form.inputbool("Display another model?")
            if not ans:
                break
        self._cluster_of_interest = int(
            input("Which cluster would you like to keep? : ")
        )
        return self

    def cluster(self) -> None:
        kmeans = KMeans(n_clusters=self._n_clusters, **self._kmeans_kwargs)
        kmeans.fit(self._scaled_features)
        self.cluster_data["Cluster"] = kmeans.labels_ + 1
        keep_trajectories = self.cluster_data[
            self.cluster_data["Cluster"] == self._cluster_of_interest
        ].index.values
        self.cutoff_df = self._df.loc[self._df["Trajectory"].isin(keep_trajectories)]
        # A minimum trajectory length is necessary for bound lifetime modeling
        self.min_length = np.min(
            self.cutoff_df.groupby(["Trajectory"])["Trajectory"].size()
        )


class Brightness:
    """Brightness thresholding"""

    def __init__(
        self,
        metadata: object,
        df: pd.DataFrame,
        method: str = None,
    ) -> None:
        """Initialize brightness object

        Args:
            metadata (class object): persistent metadata
            df (pd.DataFrame): trajectory data
            method (str, optional): cutoff approach. Defaults to None.
        """
        self.metadata = metadata
        self._df = df
        self.cutoff_df = None
        self._method = method
        self()

    @dec.Progress.start_finish(indent=2, action="Brightness", ending="Cutoffs")
    def __call__(self) -> None:
        """Calls chosen brightness thresholding method"""
        # Since the semi_auto method creates a dictionary as opposed to string
        # it requires special unpacking when called
        if "semi_auto" in self._method:
            func = getattr(
                Brightness,
                list(self._method.keys())[0],
            )
            func(
                self,
                list(self._method.values())[0][0],
                list(self._method.values())[0][1],
            )
        else:
            func = getattr(Brightness, self._method)
            func(self)

    def manual(self) -> None:
        """Manually set min and max brightness values"""
        df = self._df
        histogram = di.BrightnessHistogram(df["Average_Brightness"].unique())
        histogram.plot()
        while True:
            low_out = float(input("Select the low brightness cutoff : "))
            high_out = float(input("Select the high brightness cutoff : "))
            rm_outliers_df = df[df["Average_Brightness"].between(low_out, high_out)]
            rm_df = rm_outliers_df.reset_index(drop=True)
            print(f"Trajectories Remaining : {fo.Calc.trajectory_count(rm_df)}")
            move_on = input("Choose new cutoffs (0) or continue? (1) : ")
            if move_on == "1":
                break
        self.cutoff_df = rm_df

    def semi_auto(self, low: float, high: float) -> None:
        """
        Semi-Auto thresholding

        Set strict brightness cutoffs for all images

        Args:
            self: Brightness class object

        Returns:
            self._brightness_cutoff_df:

        Raises:
            none

        """
        df = self._df
        low_out, high_out = low, high
        rm_outliers_df = df[df["Average_Brightness"].between(low_out, high_out)]
        rm_df = rm_outliers_df.reset_index(drop=True)
        self.cutoff_df = rm_df

    def auto(self) -> None:
        """Automatically calculate brightness cutoffs"""
        df = self._df
        min_brightness = df["Average_Brightness"].min()
        max_brightness = df["Average_Brightness"].max()
        BINS = 100
        step = (max_brightness - min_brightness) / BINS
        bin_sdf = np.arange(min_brightness, max_brightness, step)
        groups = 1
        while groups <= BINS * 0.2:
            single_traj = df.drop_duplicates(subset="Trajectory", keep="first")
            sdf = (
                single_traj.groupby(
                    pd.cut(single_traj["Average_Brightness"], bins=bin_sdf)
                )
                .size()
                .nlargest(groups)
            )
            cutoff_list = np.array([i.right and i.right for i in sdf.index])
            low_out, high_out = round(np.min(cutoff_list), 3), round(
                np.max(cutoff_list), 3
            )
            rm_outliers_df = df[df["Average_Brightness"].between(low_out, high_out)]
            rm_df = rm_outliers_df.reset_index(drop=True)
            grp_df = (
                rm_df.groupby("Trajectory")
                .filter(lambda x: len(x) > self.metadata.frame_cutoff)
                .reset_index(drop=True)
            )
            # The minimum expectation for the number of trajectories can be changed
            # based on the number necessary for latter calculations
            if fo.Calc.trajectory_count(grp_df) < 150:
                groups += 1
                continue
            else:
                break
        self.cutoff_df = rm_df


class Length:
    """Remove too short of trajectories"""

    def __init__(
        self, metadata: object, df: pd.DataFrame, *, method: str = None
    ) -> None:
        """Initialize lenght object

        Args:
            metadata (class object): metadata class object
            df (pd.DataFrame): trajectory data
            method (str, optional): way to filter trajectories. Defaults to None.
        """
        self.metadata = metadata
        self._options = [
            option for option in dir(Length) if option.startswith("__") is False
        ]
        assert method in self._options, "Chosen method is not available"
        self._df = df
        self._method = method
        self.cutoff_df = None
        self()

    @dec.Progress.start_finish(indent=2, action="Length", ending="Cutoffs")
    def __call__(self) -> None:
        """Call length filtering method"""
        func = getattr(Length, self._method)
        func(self)

    def minimum(self) -> None:
        """Only use trajectories that are at least -x- frames long"""
        self.cutoff_df = (
            self._df.groupby("Trajectory")
            .filter(lambda x: len(x) > self.metadata.frame_cutoff)
            .reset_index(drop=True)
        )


class Diffusion:
    """Diffusion cutoffs"""

    def __init__(
        self,
        metadata: object,
        df: pd.DataFrame,
        *,
        low: float = 0.45,
        high: float = 3.5,
    ) -> None:
        """Initialize diffusion object

        Args:
            metadata (class object): persistent metadata
            df (pd.DataFrame): trajectory data
            low (float): low cutoff value (um^2/sec)
            high (float): high cutoff value (um^2/sec)
        """
        self.metadata = metadata
        self._df = df
        self._low = low
        self._high = high
        self.cutoff_df = None
        self()

    @dec.Progress.start_finish(indent=2, action="Diffusion", ending="Cutoffs")
    def __call__(self) -> None:
        """Call displacement cutoff function"""
        self.displacement()

    def displacement(self) -> None:
        """Use mean square displacement to filter trajectories"""
        df = self._df
        rm_outliers_df = df[
            df["MSD (\u03BCm\u00b2/sec)"].between(self._low, self._high)
        ]
        rm_df = rm_outliers_df.reset_index(drop=True)
        self.cutoff_df = rm_df
