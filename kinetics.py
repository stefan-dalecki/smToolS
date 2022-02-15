import pandas as pd
import formulas as f
import display as d
import numpy as np


class BSL:
    def __init__(self, movie, display=False):
        """
        Bound State Lifetime

        The time a molecule spends on the membrane

        Args:
            self

        Returns:
            name (string): kinetic name
            unit (string): kinetic unit

        Raises:
            none

        """

        self.name = 'BSL'
        self.unit = 'sec'
        self.x_label = f'Frames after {movie.frame_cutoff}'
        self.y_label = '% Population'
        movie.bsldf = BSL.format(movie, movie.alldf, display)

    def format(movie, df, display):
        """
        Dataframe for BSL calculations

        Generates the proper dataframe format for calculating BSL

        Args:
            self: movie class variable
            display: choose to display a scatter plot of the dataframe

        Returns:
            numpertracklen (df): dataframe for easy BSL curve fitting

        Raises:
            none

        """

        for trajectory in df['Trajectory'].unique():
            t_rows = df[df['Trajectory'] == trajectory]
            df = df.drop(t_rows.iloc[:movie.frame_cutoff].index)
        df = df.reset_index(drop=True)
        numpertracklen = pd.DataFrame(columns=['Minimum Frames', '% Tracks'])
        tot_tracks = f.Calc.traj_count(df)
        step = 2
        cutofflen = -2
        while True:
            cutofflen += step
            tracks = round(f.Calc.traj_count(
                df.groupby('Trajectory').filter(
                    lambda x: len(x) >= cutofflen).reset_index(drop=True))
                    / tot_tracks, 3)
            numpertracklen = numpertracklen.append(
                {'Minimum Frames': cutofflen,
                 '% Tracks': tracks},
                ignore_index=True)
            if tracks == 0:
                break
        if display:
            d.Scatter.scatter(numpertracklen, 'Bound State',
                              'Frames', 'Population Remaining', 
                              movie.fig_title)
        # numpertracklen.to_csv(
        #     r'E:\PhD\Data\Stefan\Best_GRP1\sample_BSL.csv', index=False)
        return numpertracklen


class MSD:
    def __init__(self):
        """
        Mean Square Displacement

        diffusion of a membrane bound monomer

        Args:
            self

        Returns:
            name (string): kinetic name
            unit (string): kinetic unit

        Raises:
            none

        """

        self.name = 'MSD'
        self.unit = '\u03BCm\u00b2/sec'
        self.x_label = None
        self.y_label = None


class RayD:
    def __init__(self, movie, list, display=False):
        """
        Rayleigh probability distribution of diffusion

        Calculate molecule diffusion of heterogeneous solutions

        Args:
            self

        Returns:
            type: description

        Raises:
            Exception: description

        """

        self.name = 'RayDifCoef'
        self.unit = '\u03BCm\u00b2/sec'
        self.x_label = None
        self.y_label = None
        movie.raydf = RayD.format(movie, list, display)

    def format(movie, list, display):
        bins = np.linspace(1/604800, 169/604800, 85)
        correct = [i+(1/604800) for i in bins][:-1]
        df = pd.DataFrame(list, columns=['values'])
        bin_data = pd.DataFrame(df.groupby(
            pd.cut(df['values'], bins=bins))
            .size().values, columns=['Frequency'])
        bin_data.index = correct
        bin_data = bin_data.reset_index(drop=False)
        if display:
            pass
        return bin_data
