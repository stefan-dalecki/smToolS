import matplotlib.pyplot as plt
import numpy as np
import formulas as f
import curvefitting as cf


class Histogram:
    """
    Histogram plots

    Display df columns as histogram

    """

    def basic(list, bins, title=None, x_label=None, y_label=None):
        """
        Basic histogram

        Histogram of binned intensity data (typically)

        Args:
            list (list): raw float data
            bins (int): number of bins
            title (str): figure title
            x_label (str): x-axis label
            y_label (str): y-axis label

        Returns:
            histogram

        Raises:
            none

        """

        plt.hist(list, bins, range=[min(list), max(list)],
                 edgecolor='white', linewidth=1, color='black')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.grid(axis='x', color='grey')
        plt.tight_layout()
        plt.show(block=True)

    def lines(list, bins, low_cut, high_cut,
              title=None, x_label=None, y_label=None):
        """
        Histogram with lines

        Vertical lines designate cutoffs for downstream removal

        Args:
            list (list): raw float data
            bins (int): number of bins
            low_cut (float): low intensity cutoff
            high_cut (float): high intensity cutoff
            title (str): figure title
            x_label (str): x-axis label
            y_label (str): y-axis label

        Returns:
            histogram with vertical lines

        Raises:
            none

        """

        n, bins, patches = plt.hist(list, bins, range=[min(list), max(list)],
                                    edgecolor='white', linewidth=2,
                                    color='black')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.vlines(low_cut, 0, n.max(), colors='mediumorchid', linewidth=2)
        plt.vlines(high_cut, 0, n.max(), colors='mediumorchid', linewidth=2)
        plt.tight_layout()
        plt.show(block=True)


class Scatter:
    """
    Scatterplot

    scatterplots with optional line overlays


    """

    def scatter(df, title=None, x_label=None, y_label=None):
        """
        Basic scatterplot

        Uses a two column dataframe to create scatter

        Args:
            df (df): two column dataframe, x and y values in 1st/2nd columns
            title (str): figure title
            x_label (str): x-axis label
            y_label (str): y-axis label

        Returns:
            scatterplot

        Raises:
            AssertionError: dataframe lacks at least two columns

        """

        assert df.shape[1] >= 2, 'Dataframe doesn not have at least 2 columns'
        x = df.iloc[:, 0].values
        y = df.iloc[:, 1].values
        plt.scatter(x, y, s=2, alpha=1, color='black')
        plt.title(title)
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


class ExpDecay:
    """
    Exponential decay functions

    Scatterplot with overlayed line

    """

    def onecomp(movie, df, equation, tau1, cov1, r2, kinetic,
                title=None, x_label=None, y_label=None):
        x_data = df.iloc[:, 0].values.astype(float)
        y_data = df.iloc[:, 1].values.astype(float)
        val1 = np.round(tau1*movie.framestep_size, 3)
        var1 = np.format_float_scientific(cov1, precision=1, exp_digits=2)
        r_val = round(r2, 6)
        plt.scatter(x_data, y_data, s=2, alpha=1, color='black',
                    label=f'Data: n = {f.Calc.traj_count(movie.df)}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title + '--- One Comp Exp Decay ---')
        plt.plot(x_data, equation(x_data, tau1),
                 label=f'{kinetic.name}: {val1} {kinetic.unit}\n'
                 + f' Cov: {var1}\n'
                 + f'R\u00b2: {r_val}',
                 linestyle='dashed',
                 color='mediumorchid', linewidth=2, alpha=0.9)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def twocomp(movie, df, a, tau1, cov1, tau2, cov2, r2, kinetic,
                title=None, x_label=None, y_label=None):
        x_data = df.iloc[:, 0].values.astype(float)
        y_data = df.iloc[:, 1].values.astype(float)
        a1 = round(a, 3)*100
        a2 = round(100-a1, 3)
        val1 = round(tau1*movie.framestep_size, 3)
        val2 = round(tau2*movie.framestep_size, 3)
        var1 = np.format_float_scientific(cov1, precision=1, exp_digits=2)
        var2 = np.format_float_scientific(cov2, precision=1, exp_digits=2)
        r_val = round(r2, 6)
        plt.scatter(x_data, y_data, s=2, alpha=1, color='black',
                    label=f'Data: n = {f.Calc.traj_count(movie.df)}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(movie.fig_title + '--- Two Comp Exp Decay ---')
        plt.plot(x_data, cf.TwoCompExpDecay.equation(
            x_data, a, tau1, tau2),
            label=f'Maj Frac: {a1}\n'
            + f' {kinetic.name}: {val1} {kinetic.unit}\n'
            + f' Cov {var1}\n'
            + f'Min Frac: {a2}\n'
            + f' {kinetic.name}: {val2} {kinetic.unit}\n'
            + f' Cov: {var2}\n'
            + f'R\u00b2: {r_val}',
            linestyle='dashed',
            color='mediumorchid', linewidth=2, alpha=0.9)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()

    def explin(movie, df, tau1, cov1, b, cov2, r2, kinetic,
               title=None, x_label=None, y_label=None):
        x_data = df.iloc[:, 0].values.astype(float)
        y_data = df.iloc[:, 1].values.astype(float)
        val1 = round(tau1*movie.framestep_size, 4)
        val2 = np.format_float_scientific(b*movie.framestep_size,
                                          precision=1, exp_digits=2)
        var1 = np.format_float_scientific(cov1, precision=1, exp_digits=2)
        var2 = np.format_float_scientific(cov2, precision=1, exp_digits=2)
        r_val = round(r2, 6)
        plt.scatter(x_data, y_data, s=2, alpha=1, color='black',
                    label=f'Data: n = {f.Calc.traj_count(movie.df)}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title + '--- Exp and Lin Decay ---')
        plt.plot(x_data, cf.ExpLinDecay.equation(
            x_data, tau1, b),
            label=f'Exp {kinetic.name}: {val1} {kinetic.unit}\n'
            + f' Exp Cov:{var1}\n'
            + f'Lin {kinetic.name}: {val2} frames\n'
            + f' Lin Cov: {var2}\n'
            + f'R\u00b2: {r_val}',
            linestyle='dashed',
            color='mediumorchid', linewidth=2, alpha=0.9)
        plt.legend(loc='upper right')
        plt.tight_layout()
        plt.show()


class Rayleigh:
    def onecomp(movie, df, a, sig, sig_cov, r2, kinetic,
                       title=None, x_label=None, y_label=None):
        x_data = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values
        raydiffcoeff = round(1e8*sig**2/(2*movie.framestep_size), 4)
        varsig = np.format_float_scientific(
            sig_cov, precision=1, exp_digits=2)
        r_val = round(r2, 6)
        plt.scatter(x_data, y_data, s=4, alpha=1, color='black',
                    label=f'Data: n = {f.Calc.traj_count(movie.df)}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title + '--- One Component Rayleigh ---')
        plt.plot(x_data, cf.OneCompRayleigh.equation(
            x_data, a, sig),
            label=f'{kinetic.name}: {raydiffcoeff} {kinetic.unit}\n'
            + f' Cov {varsig}\n'
            + f'R\u00b2: {r_val}',
            linestyle='dashed',
            color='mediumorchid', linewidth=2, alpha=0.9)
        plt.legend(loc='upper right')
        plt.show()

    def twocomp(movie, df, a, b, sig1, sig1_cov, sig2, sig2_cov, r2, kinetic,
                       title=None, x_label=None, y_label=None):
        x_data = df.iloc[:, 0].values
        y_data = df.iloc[:, 1].values
        raypop1, raypop2 = round(a/(a+b), 4), round(b/(a+b), 4)
        raydiffcoeff1 = round(1e8*sig1**2/(2*movie.framestep_size), 4)
        raydiffcoeff2 = round(1e8*sig2**2/(2*movie.framestep_size), 4)
        varsig1 = np.format_float_scientific(
            sig1_cov, precision=1, exp_digits=2)
        varsig2 = np.format_float_scientific(
            sig2_cov, precision=1, exp_digits=2)
        r_val = round(r2, 6)
        plt.scatter(x_data, y_data, s=4, alpha=1, color='black',
                    label=f'Data: n = {f.Calc.traj_count(movie.df)}')
        plt.xlabel(x_label)
        plt.ylabel(y_label)
        plt.title(title + '--- Two Component Rayleigh ---')
        plt.plot(x_data, cf.TwoCompRayleigh.equation(
            x_data, a, b, sig1, sig2),
            label=f'Maj Frac: {raypop1}\n'
            + f' {kinetic.name}: {raydiffcoeff1} {kinetic.unit}\n'
            + f' Cov {varsig1}\n'
            + f'Min Frac: {raypop2}\n'
            + f' {kinetic.name}: {raydiffcoeff2} {kinetic.unit}\n'
            + f' Cov: {varsig2}\n'
            + f'R\u00b2: {r_val}',
            linestyle='dashed',
            color='mediumorchid', linewidth=2, alpha=0.9)
        plt.legend(loc='upper right')
        plt.show()
