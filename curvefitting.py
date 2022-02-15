import formulas as f
import numpy as np
import display as d
from scipy.optimize import curve_fit
from scipy import stats


class OneCompExpDecay:
    """
    One component exponential decay analyses

    Workflow for fitting including equation, bounds, outputs, and figure

    """

    def __init__(self, movie, kinetic, display=False):
        """
        Full model fitting process

        Calling this function runs a complete model fitting series

        Args:
            self: placeholder variable
            movie: movie class object
            kinetic: kinetic of interest
            display (bool): choice to display the resulting fit data

        Returns:
            outputs data to movie class object

        Raises:
            none

        """

        estimation = f.Calc.e_estimation(movie.bsldf)
        popt, pcov, r2, sumsqr = FitFunctions.curve(movie, movie.bsldf,
                                                    OneCompExpDecay.equation,
                                                    [estimation],
                                                    OneCompExpDecay.bounds(),
                                                    OneCompExpDecay.output,
                                                    kinetic)
        if display:
            OneCompExpDecay.figure(movie, movie.bsldf, popt, pcov, r2, kinetic)

    def equation(t, tau):
        """
        One component equation

        e^(-t / tau)

        Args:
            t (int): time
            tau (float): decay time constant

        Returns:
            single exponential decay equation

        Raises:
            none

        """

        return np.exp(-t / tau)

    def bounds():
        """
        One component bounds

        no bounds are set for time constant tau

        Args:
            none

        Returns:
            list: min and max parameter values

        Raises:
            none

        """

        return [(-np.inf), (np.inf)]

    def output(movie, popt, pcov, r2, kinetic):
        """
        One component exponential decay outputs

        Outputs are selected based of the kinetic chosen

        Args:
            self: movie class object
            popt (list): parameter values
            pcov (list): parameter covariance
            r2 (float): goodness of fit
            kinetic: input kinetic class

        Returns:
            none

        Raises:
            none

        """

        movie.exportdata.update({f'OneExp {kinetic.name} ({kinetic.unit})':
                                popt[0]*movie.framestep_size,
                                f'OneExp {kinetic.name} Cov': pcov[0],
                                 'R\u00b2 Exp': r2})

    def figure(movie, df, popt, pcov, r2, kinetic):
        """
        Single exponential decay figure

        Overlays fit line onto a decay scatter plot

        Args:
            movie: movie class object
            df: dataframe for scatter plot
            popt (list): parameter values
            pcov (list): parameter covariance
            r2 (float): goodness of fit
            kinetic: kinetic class identity

        Returns:
            scatter plot overlayed with line

        Raises:
            none

        """

        d.ExpDecay.onecomp(movie, df, OneCompExpDecay.equation,
                           popt[0], pcov[0], r2, kinetic,
                           title=movie.fig_title,
                           x_label=kinetic.x_label,
                           y_label=kinetic.y_label)


class TwoCompExpDecay:
    """
    Two component exponential decay analyses

    Workflow for fitting including equation, bounds, outputs, and figure

    """

    def __init__(self, movie, kinetic, display=False):
        """
        Full model fitting process

        Calling this function runs a complete model fitting series

        Args:
            self: placeholder variable
            movie: movie class object
            kinetic: kinetic of interest
            display: choice to display the resulting fit data

        Returns:
            outputs data to movie class object

        Raises:
            none

        """

        estimation = f.Calc.e_estimation(movie.bsldf)
        popt, pcov, r2, ssqr = FitFunctions.curve(movie, movie.bsldf,
                                                  TwoCompExpDecay.equation,
                                                  [1, estimation, estimation],
                                                  TwoCompExpDecay.bounds(),
                                                  TwoCompExpDecay.output,
                                                  kinetic)
        if display:
            TwoCompExpDecay.figure(movie, movie.bsldf, popt, pcov, r2, kinetic)

    def equation(t, a, tau1, tau2):
        """
        Two component equation

        a * e^(-t / tau1) + (1-a) * e^(-t / tau2)

        Args:
            t (int): time
            a (float): percent population
            tau1 (float): one decay time constant
            tau2 (float): other decay time constant

        Returns:
            double exponential decay equation

        Raises:
            none

        """

        return a*np.exp(-t / tau1) + (1-a)*np.exp(-t / tau2)

    def bounds():
        """
        Two component bounds

        a must be between 0 and 1
        no bounds are set for time constants

        Args:
            none

        Returns:
            list: min and max parameter values

        Raises:
            none

        """

        return [(0, -np.inf, -np.inf), (1, np.inf, np.inf)]

    def output(self, popt, pcov, r2, kinetic):
        """
        Two component exponential decay outputs

        Outputs are selected based of the kinetic chosen
        corrections so that major fraction is always first

        Args:
            self: movie class object
            popt (list): parameter values
            pcov (list): parameter covariance
            r2 (float): goodness of fit
            kinetic: input kinetic class

        Returns:
            none

        Raises:
            none

        """

        if popt[0] < 0.5:
            popt[0] = 1 - popt[0]
            popt[1], popt[2] = popt[2], popt[1]
            pcov[1], pcov[2] = pcov[2], pcov[1]
        self.exportdata.update({'TwoExpA Maj Frac (%)': popt[0]*100,
                                'TwoExpA Maj Frac Cov': pcov[0],
                                f'TwoExpA {kinetic.name} Maj ({kinetic.unit})':
                                popt[1]*self.framestep_size,
                                f'TwoExpA {kinetic.name} Maj Cov': pcov[1],
                                f'TwoExpA {kinetic.name} Min Frac':
                                100-popt[0]*100,
                                f'TwoExpA {kinetic.name} Min ({kinetic.unit})':
                                popt[2]*self.framestep_size,
                                f'TwoExpA {kinetic.name} Min Cov': pcov[2],
                                'R\u00b2 ExpExpA': r2})

    def figure(movie, df, popt, pcov, r2, kinetic):
        """
        Double exponential decay figure

        Overlays fit line onto a decay scatter plot

        Args:
            self: movie class object
            df (df): dataframe for scatter plot
            popt (list): parameter values
            pcov (list): parameter covariance
            r2 (float): goodness of fit
            kinetic: kinetic class identity

        Returns:
            scatter plot overlayed with line

        Raises:
            none

        """

        d.ExpDecay.twocomp(movie, df, popt[0], popt[1], pcov[1],
                           popt[2], pcov[2], r2, kinetic,
                           title=movie.fig_title,
                           x_label=kinetic.x_label,
                           y_label=kinetic.y_label)


class Alt_TwoCompExpDecay:
    """
    Alternative two component exponential decay analyses

    Workflow for fitting including equation, bounds, outputs, and figure
    replaces 'a' in equation with two separate population variables

    """

    def __init__(self, movie, kinetic, display=False):
        """
        Full model fitting process

        Calling this function runs a complete model fitting series

        Args:
            self: placeholder variable
            movie: movie class object
            kinetic: kinetic of interest
            display: choice to display the resulting fit data

        Returns:
            outputs data to movie class object

        Raises:
            none

        """

        estimation = f.Calc.e_estimation(movie.bsldf)
        popt, pcov, r2, ssqr = FitFunctions.curve(movie, movie.bsldf,
                                                  Alt_TwoCompExpDecay.equation,
                                                  [0.5, 0.5, estimation,
                                                   estimation],
                                                  Alt_TwoCompExpDecay.bounds(),
                                                  Alt_TwoCompExpDecay.output,
                                                  kinetic)
        if display:
            Alt_TwoCompExpDecay.figure(
                movie, movie.bsldf, popt, pcov, r2, kinetic)

    def equation(t, a, b, tau1, tau2):
        """
        Alt two component equation

        a * e^(-t / tau1) + b * e^(-t / tau2)

        Args:
            t (interest): time
            a (float): magnitude of one population
            b (float): mangitude of other population
            tau1 (float): one decay time constant
            tau2 (float): other decay time constant

        Returns:
            alternate double exponential decay equation

        Raises:
            none

        """

        return a*np.exp(-t / tau1) + b*np.exp(-t / tau2)

    def bounds():
        """
        Alt two component bounds

        a and b cannot be negative
        no bounds are set for time constants

        Args:
            none

        Returns:
            list: min and max parameter values

        Raises:
            none

        """

        return [(0, 0, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf)]

    def output(movie, popt, pcov, r2, kinetic):
        """
        Alt two component exponential decay outputs

        Outputs are selected based of the kinetic chosen
        corrections so that the larger magnitude constant is always first

        Args:
            self: movie class object
            popt (list): parameter values
            pcov (list): parameter covariance
            r2 (float): goodness of fit
            kinetic: input kinetic class

        Returns:
            none

        Raises:
            none

        """

        if popt[0] < popt[1]:
            popt[0], popt[1] = popt[1], popt[0]
            pcov[0], pcov[1] = pcov[1], pcov[0]
            popt[2], popt[3] = popt[3], popt[2]
            pcov[2], pcov[3] = pcov[3], pcov[2]
        movie.exportdata.update({'Alt_TwoExp Maj Frac (%)':
                                100*popt[0]/(popt[0]+popt[1]),
                                'Alt_TwoExp Maj Frac Cov': pcov[0],
                                 f'Alt_TwoExp Maj {kinetic.name}'
                                 f'({kinetic.unit})':
                                 popt[2]*movie.framestep_size,
                                 f'Alt_TwoExp Maj {kinetic.name} Cov': pcov[2],
                                 'Alt_TwoExp Min Frac (%)':
                                 100*popt[1]/(popt[0]+popt[1]),
                                 'Alt_TwoExp Min Frac Cov': pcov[1],
                                 f'Alt_TwoExp Min {kinetic.name}'
                                 f'({kinetic.unit})':
                                 popt[3]*movie.framestep_size,
                                 f'Alt_TwoExp Min {kinetic.name} Cov': pcov[3],
                                 'R\u00b2 Alt_TwoExp': r2})

    def figure(movie, df, popt, cov, r2, kinetic):
        """
        Alt double exponential decay figure

        Overlays fit line onto a decay scatter plot

        Args:
            self: movie class object
            df (df): dataframe for scatter plot
            popt (list): parameter values
            pcov (list): parameter covariance
            r2 (float): goodness of fit
            kinetic: kinetic class identity

        Returns:
            scatter plot overlayed with line

        Raises:
            none

        """

        d.ExpDecay.twocomp(movie, df,
                           popt[0]/(popt[0]+popt[1]),
                           popt[2], cov[2],
                           popt[3], cov[3],
                           r2, kinetic,
                           title=movie.fig_title,
                           x_label=kinetic.x_label,
                           y_label=kinetic.y_label)


class ExpLinDecay:
    """
    Exponential decay with linear component

    Workflow for fitting including equation, bounds, outputs, and figure

    """

    def __init__(self, movie, kinetic, display=False):
        """
        Full model fitting process

        Calling this function runs a complete model fitting series

        Args:
            self: placeholder variable
            movie: movie class object
            kinetic: kinetic of interest
            display: choice to display the resulting fit data

        Returns:
            outputs data to movie class object

        Raises:
            none

        """

        estimation = f.Calc.e_estimation(movie.bsldf)
        popt, pcov, r2, sumqr = FitFunctions.curve(movie, movie.bsldf,
                                                   ExpLinDecay.equation,
                                                   [estimation, 0.3],
                                                   ExpLinDecay.bounds(),
                                                   ExpLinDecay.output,
                                                   kinetic)
        if display:
            ExpLinDecay.figure(movie, movie.bsldf, popt, pcov, r2, kinetic)

    def equation(t, tau, b):
        """
        Exponential and linear equation

        e^(-t / tau) + b*x

        Args:
            t (int): time
            tau (float): exponential time constant
            b (float): linear component slope

        Returns:
            exponential and linear equation

        Raises:
            none

        """

        return np.exp(-t / tau) + b*t

    def bounds():
        """
        exp and lin bounds

        no restrictions on tau or b

        Args:
            none

        Returns:
            list: min and max parameter values

        Raises:
            none

        """

        return [(-np.inf, -np.inf), (np.inf, np.inf)]

    def output(movie, popt, pcov, r2, kinetic):
        """
        Exponential and linear model output

        Outputs are selected based of the kinetic chosen

        Args:
            self: movie class object
            popt (list): parameter values
            pcov (list): parameter covariance
            r2 (float): goodness of fit
            kinetic: input kinetic class

        Returns:
            none

        Raises:
            none

        """

        movie.exportdata.update({f'Exp {kinetic.name} ({kinetic.unit})':
                                popt[0]*movie.framestep_size,
                                f'Exp {kinetic.name} Cov': pcov[0],
                                 'Lin Comp (frames)': popt[1],
                                 'Lin Comp Cov': pcov[1],
                                 'R\u00b2 ExpLin': r2})

    def figure(movie, df, popt, cov, r2, kinetic):
        """
        Exponential and linear model figure

        Overlays fit line onto a decay scatter plot

        Args:
            self: movie class object
            df (df): dataframe for scatter plot
            popt (list): parameter values
            pcov (list): parameter covariance
            r2 (float): goodness of fit
            kinetic: kinetic class identity

        Returns:
            scatter plot overlayed with line

        Raises:
            none

        """

        d.ExpDecay.explin(movie, df,
                          popt[0], cov[0],
                          popt[1], cov[1],
                          r2, kinetic,
                          title=movie.fig_title,
                          x_label=kinetic.x_label,
                          y_label=kinetic.y_label)


# Two Part Rayleigh (a*Rayleigh Probability Density Fuction + b*...)
class TwoCompRayleigh:
    """
    Two component Rayleigh fitting workflow

    Workflow for fitting including equation, bounds, outputs, and figure
    ***incomplete***

    """

    def __init__(self, movie, kinetic, display=False):
        """
        Full model fitting process

        Calling this function runs a complete model fitting series

        Args:
            self: placeholder variable
            movie: movie class object
            kinetic: kinetic of interest
            display: choice to display the resulting fit data

        Returns:
            outputs data to movie class object

        Raises:
            none

        """

        estimation = [3e-3, 3e-3, 1e-5, 1e-5]
        popt, pcov, r2, sumsqr = FitFunctions.curve(movie, movie.raydf,
                                                    TwoCompRayleigh.equation,
                                                    estimation,
                                                    TwoCompRayleigh.bounds(),
                                                    TwoCompRayleigh.output,
                                                    kinetic)

    def equation(t, a, b, sig1, sig2):
        """
        Exponential and linear equation

        a * rayleigh probability distribution +
        b * rayleigh probability distribution

        Args:
            t (int): time
            a (float): magnitude of one population
            b (float): mangitude of other population
            sig1 (float): one diffusion constant
            sig2 (float): other diffusion constant

        Returns:
            two component Rayleigh equation

        Raises:
            none

        """

        return (a*(t*np.exp(-t**2/(2*sig1**2)))/sig1**2) \
            + (b*(t*np.exp(-t**2/(2*sig2**2)))/sig2**2)

    def bounds():
        """
        Two component Rayleigh bounds

        a and b cannot be negative
        no bounds are set for diffusion constants

        Args:
            none

        Returns:
            list: min and max parameter values

        Raises:
            none

        """

        return [(0, 0, -np.inf, -np.inf), (np.inf, np.inf, np.inf, np.inf)]

    def output(movie, popt, cov, r2, kinetic):
        if popt[0] < popt[1]:
            popt[0], popt[1] = popt[1], popt[0]
            cov[0], cov[1] = cov[1], cov[0]
            popt[2], popt[3] = popt[3], popt[2]
            cov[2], cov[3] = cov[3], cov[2]
        movie.exportdata.update({f'{kinetic.name} Maj Frac (%)':
                                popt[0]/(popt[0]+popt[1])*100,
                                f'{kinetic.name} Maj Frac Cov': cov[0],
                                 f'{kinetic.name} ({kinetic.unit})':
                                 1e8*popt[2]**2/(2*movie.framestep_size),
                                 f'{kinetic.name} Maj Cov': cov[2],
                                 f'{kinetic.name} Min Frac (%)':
                                 popt[1]/(popt[0]+popt[1])*100,
                                 f'{kinetic.name} Min Frac Cov': cov[1],
                                 f'{kinetic.name} Min ({kinetic.unit})':
                                 1e8*popt[3]**2/(2*movie.framestep_size),
                                 f'{kinetic.name} Min Cov': cov[3],
                                 'R\u00b2 Ray': r2})


class FitFunctions:
    """
    Various methods for fitting data

    Fit curved, linear, and Rayleigh distribution data
    Rayleigh distributions not currently opperational

    """

    def curve(movie, df, equation, p0, limits, output_method, kinetic,
              method='trf'):
        """
        Curve fitting using non linear least squares

        fit scatter plot data using a selection of functions

        Args:
            self: movie class objects
            df (df): dataframe with two columns, x and y
            equation: equation with parameters used to fit dataframe
            p0 (list): initial guesses for parameters
            limits (list): parameter boundaries
            output_method: how data is added to the movie.df attribute
            kinetic: kinetic being modeld
            method: options include 'trf', 'dogbox', and 'lm'
                'lm' should not be used with boundaries

        Returns:
            type: description

        Raises:
            Exception: description

        """
        try:
            x_data = df.iloc[:, 0].values
            y_data = df.iloc[:, 1].values
            if limits is None:
                popt, pcov = curve_fit(equation, x_data, y_data,
                                       p0=p0)
            else:
                popt, pcov = curve_fit(equation, x_data, y_data,
                                       p0=p0, bounds=limits)
            popt, pcov = curve_fit(equation, x_data, y_data,
                                   p0=p0, bounds=limits)
            pcov = (np.sqrt(np.diag(pcov)))
            res = y_data - equation(x_data, *popt)
            sum_sqr_res = np.sum(res**2)
            sum_sqr_tot = np.sum((y_data-np.mean(y_data))**2)
            r2 = 1 - (sum_sqr_res / sum_sqr_tot)
            output_method(movie, popt, pcov, r2, kinetic)
            return popt, pcov, r2, sum_sqr_res
        except RuntimeError:
            return np.nan, np.nan, np.nan, np.nan

    def linear(movie, df, x_col, y_col, kinetic):
        """
        Linear regression method (y = m*x + b)

        Fit a line on a 2D set of data
        Contains ability to select exact columns of interest

        Args:
            self: movie class object
            df (df): contains all data
            x_col (string): x values column name
            y_col (string): y values column name
            kinetic: kinetic class object

        Returns:
            slope (float): line slope
            intercept (float): y-axis intercept
            r2: goodness of fit
            p: p-value whether the slope is zero
            se: standard error of slope

        Raises:
            none

        """

        x_data = df[x_col].values
        y_data = df[y_col].values
        slope, intercept, r2, p, se = stats.linregress(x_data, y_data)
        # tinv = lambda p, degf: abs(t.ppf(p/2, degf))
        # ts = tinv(0.05, len(x_data)-2)
        movie.exportdata.update({f'{kinetic.name} ({kinetic.unit})':
                                slope*10**8,
                                f'{kinetic.name} R\u00b2:': r2})
        return slope, intercept, r2, p, se
