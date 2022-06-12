"""Filter trajectory data"""

from collections import defaultdict
from statistics import mean
from copy import deepcopy
import numpy as np
import pandas as pd
from kneed import KneeLocator
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import display as di
import formulas as fo


class Clustering:
    """K-means clustering thresholding"""

    class Clustering:
        """K-means clustering thresholding"""

    def __init__(self, metadata, df: pd.DataFrame) -> None:

        """Initialize clustering object

        Args:
            df (pd.DataFrame): raw trajectory data
        """
        self.metadata = metadata
        self._df = df
        self.cluster_data = None
        self._n_clusters = None
        self._kmeans_kwargs = None
        self.suggested_clusters = {"Elbow": None, "Silhouette": None}
        self.onestep_SDs = []
        self._scaled_features = None
        self._cluster_of_interest = None
        self.min_length = None
        self.cutoff_df = None

    def add_parameters(self) -> None:
        brightnesses = []
        lengths = []
        MSDs = []

        for traj in self._df["Trajectory"].unique():
            t_rows = self._df[self._df["Trajectory"] == traj]
            length = t_rows.shape[0]
            lengths.append(length)
            brightness = np.mean(t_rows["Brightness"])
            brightnesses.append(brightness)
            x_col = t_rows["x"].values
            y_col = t_rows["y"].values
            SDs = []
            for step in range(len(t_rows) - 1):
                x1, y1 = x_col[step], y_col[step]
                x2, y2 = x_col[step + 1], y_col[step + 1]
                squared_distance = fo.Calc.distance(x1, y1, x2, y2) ** 2
                SDs.append(squared_distance)
            self.onestep_SDs += SDs
            MSDs.append(
                np.mean(SDs)
                * self.metadata.pixel_size**2
                / (4 * self.metadata.framestep_size)
                * 1e8
            )
        self.cluster_data = pd.DataFrame(
            {
                "Average_Brightness": brightnesses,
                "Length (frames)": lengths,
                "Diffusion (\u03BCm\u00b2/sec)": MSDs,
            }
        )
        self.cluster_data.index += 1
        scaler = StandardScaler()
        self._scaled_features = scaler.fit_transform(self.cluster_data)
        return self

    def estimate_clusters(self) -> None:
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
        return self

    def display(self) -> None:
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
        self.min_length = np.min(
            self.cutoff_df.groupby(["Trajectory"])["Trajectory"].size()
        )


class Brightness:
    """Brightness thresholding"""

    def __init__(
        self,
        metadata,
        df: pd.DataFrame,
        method=None,
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

    def __call__(self) -> None:
        """Calls chosen brightness thresholding method"""
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
        while True:
            di.BrightnessHistogram(
                df.groupby("Trajectory")["Brightness"].mean().values
            ).plot()
            low_out = float(input("Select the low brightness cutoff : "))
            high_out = float(input("Select the high brightness cutoff : "))
            not_low = df[df["Brightness-Average"] > low_out]
            not_high = df[df["Brightness-Average"] < high_out]
            list_low = not_low["Trajectory"].values
            list_high = not_high["Trajectory"].values
            rm_outliers_df = df[df["Trajectory"].isin(list_low)]
            rm_outliers_df = rm_outliers_df[
                rm_outliers_df["Trajectory"].isin(list_high)
            ]
            rm_df = rm_outliers_df.reset_index(drop=True)
            print(fo.Calc.trajectory_count(rm_df))
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
        not_low = df[df["Brightness-Average"] > low_out]
        not_high = df[df["Brightness-Average"] < high_out]
        list_low = not_low["Trajectory"].values
        list_high = not_high["Trajectory"].values
        rm_outliers_df = df[df["Trajectory"].isin(list_low)]
        rm_outliers_df = rm_outliers_df[rm_outliers_df["Trajectory"].isin(list_high)]
        rm_df = rm_outliers_df.reset_index(drop=True)
        self.cutoff_df = rm_df

    def auto(self) -> None:
        """Automatically calculate brightness cutoffs"""
        df = self._df
        min_brightness = df["Brightness-Average"].min()
        max_brightness = df["Brightness-Average"].max()
        BINS = 100
        step = (max_brightness - min_brightness) / BINS
        bin_sdf = np.arange(min_brightness, max_brightness, step)
        groups = 1
        while groups <= BINS * 0.2:
            single_traj = df.drop_duplicates(subset="Trajectory", keep="first")
            sdf = (
                single_traj.groupby(
                    pd.cut(single_traj["Brightness-Average"], bins=bin_sdf)
                )
                .size()
                .nlargest(groups)
            )
            cutoff_list = np.array([i.right and i.right for i in sdf.index])
            low_out, high_out = round(np.min(cutoff_list), 3), round(
                np.max(cutoff_list), 3
            )
            not_low = df[df["Brightness-Average"] > low_out]
            not_high = df[df["Brightness-Average"] < high_out]
            list_low = not_low["Trajectory"].values
            list_high = not_high["Trajectory"].values
            rm_outliers_df = df[df["Trajectory"].isin(list_low)]
            rm_outliers_df = rm_outliers_df[
                rm_outliers_df["Trajectory"].isin(list_high)
            ]
            rm_df = rm_outliers_df.reset_index(drop=True)
            grp_df = (
                rm_df.groupby("Trajectory")
                .filter(lambda x: len(x) > self.metadata.frame_cutoff)
                .reset_index(drop=True)
            )
            if fo.Calc.trajectory_count(grp_df) > 150:
                break
            else:
                groups += 1
                continue
        self.cutoff_df = rm_df


class Length:
    """Remove too short of trajectories"""

    def __init__(self, metadata, df: pd.DataFrame, *, method: str = None) -> None:
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

    def subtract(self) -> None:
        """Remove -x- frames from the beginning of all trajectories"""
        df = self._df
        for trajectory in df["Trajectory"].unique():
            t_rows = df[df["Trajectory"] == trajectory]
            df.drop(t_rows.iloc[: self.metadata.frame_cutoff].index)
        self.cutoff_df = df.reset_index(drop=True)


class Diffusion:
    """Diffusion cutoffs"""

    def __init__(self, metadata, df: pd.DataFrame) -> None:
        """Initialize diffusion object

        Args:
            metadata (class object): persistent metadata
            df (pd.DataFrame): trajectory data
        """
        self.metadata = metadata
        self._df = df
        self.cutoff_df = None
        self.mean_SDs = None
        self.onestep_SDs = None
        self()

    def __call__(self) -> None:
        """Call displacement cutoff function"""
        self.displacement()

    def displacement(self) -> None:
        """Use mean square displacement to filter trajectories"""
        df = self._df
        all_SDs = defaultdict(list)
        mean_SDs = pd.DataFrame(columns=["Step Length", "MSD"])
        onestep_SDs = []
        for trajectory in df["Trajectory"].unique():
            t_rows = df[df["Trajectory"] == trajectory]
            if len(t_rows) > 4:
                SD_SL1 = []
                x_col = t_rows["x"].values
                y_col = t_rows["y"].values
                if len(t_rows - 3) > 8:
                    max_step_len = 8
                else:
                    max_step_len = len(t_rows - 3)
                for step_len in range(1, max_step_len + 1):
                    for step_num in range(0, len(t_rows) - step_len - 1):
                        x1, y1 = x_col[step_num], y_col[step_num]
                        x2, y2 = x_col[step_num + step_len], y_col[step_num + step_len]
                        squared_distance = fo.Calc.distance(x1, y1, x2, y2) ** 2
                        if step_len == 1:
                            SD_SL1.append(squared_distance)
                    if step_len == 1:
                        diff_coeff1 = (
                            mean(SD_SL1)
                            * self.metadata.pixel_size**2
                            / (4 * self.metadata.framestep_size)
                        )
                        if diff_coeff1 <= 1e-9 or diff_coeff1 >= 3e-8:
                            df.drop(
                                df.loc[df["Trajectory"] == trajectory].index,
                                inplace=True,
                            )
                            break
                        else:
                            onestep_SDs += [
                                i ** (1 / 2) * self.metadata.pixel_size for i in SD_SL1
                            ]
                            all_SDs[step_len].append(squared_distance)
                    else:
                        all_SDs[step_len].append(squared_distance)
        for i in range(1, max_step_len + 1):
            mean_steplen = (
                mean(all_SDs[i])
                * self.metadata.pixel_size**2
                / (4 * self.metadata.framestep_size)
            )
            new_row = pd.DataFrame.from_dict(
                {"Step Length": [i], "MSD": [mean_steplen]}
            )
            mean_SDs = pd.concat([mean_SDs, new_row])
        self.mean_SDs = mean_SDs
        self.onestep_SDs = onestep_SDs
        self.cutoff_df = df
