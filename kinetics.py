import pandas as pd
import formulas as f
import display as d
import numpy as np


class BSL:
    """
    Bound state lifetime

    The time a molecule spends on the membrane

    """

    def __init__(self, movie, display=False):
        """
        BSL key features

        attributes for later processing

        Args:
            self
            movie: movie class object
            display (bool): choice of whether to display figures

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
            df (df): dataframe to generate BSL df
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
    """
    Mean Square Displacement

    diffusion of a membrane bound monomer

    """

    def __init__(self):
        """
        MSD key features

        establish key metadata pieces

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
    """
    Rayleigh probability distribution of diffusion

    Calculate molecule diffusion of homogenous and heterogeneous solutions

    """

    def __init__(self, movie, list, display=False):
        """
        Rayleigh metrics

        set Rayleigh kinetic metadata

        Args:
            self
            movie: movie class objects
            list (list): data to generate Rayleigh df

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
        """
        Dataframe for Rayleigh diffusion calculations

        Bin data into sizes necessary for rayleigh PDF overlays

        Args:
            movie: movie class objects
            list (list): one step diffusion coefficient values
            display (bool): choice to display the figure

        Returns:
            type: description

        Raises:
            Exception: description

        """

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
