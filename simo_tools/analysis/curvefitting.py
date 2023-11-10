"""
Fit kinetic data to models Once kinetic (kinetics.py) raw data is organized, K-off (BSL)
and other features can be extracted.
"""

import copy
import itertools
import logging
from abc import ABCMeta, abstractmethod
from inspect import signature
from typing import Dict, Self, Type, Union

import numpy as np
import pandas as pd
from num2words import num2words
from scipy import stats
from scipy.optimize import curve_fit
from smToolS import metadata
from smToolS.analysis_tools import kinetics as kin

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def e_estimation(df: pd.DataFrame) -> float:
    """
    Estimate decay time constant.

    Args:
        df (pd.DataFrame): formatted BSL kinetic table

    Returns:
        float: 1/e estimation of time constant

    """

    estimation = df.iloc[(df.iloc[:, 1] - (1 / np.exp(1))).abs().argsort()[:1]]
    estimation = estimation.iloc[:, 0].values[0]
    return estimation


def ordinal(number: int) -> str:
    """
    Convert integer to ordinal string.

    Args:
        number (int): integer

    Returns:
        str: ordinal string (i.e. 1st, 2nd, 89th, ...)

    """
    return "%d%s" % (
        number,
        "tsnrhtdd"[(number / 10 % 10 != 1) * (number % 10 < 4) * (number % 10) :: 4],
    )


class Model:
    """
    Model template.
    """

    def __init__(self) -> None:
        """
        Instantiate the empty model object.
        """
        self.microscope = None
        self.movie = None
        self.kinetic = None
        self.model_name = None
        self.equation = None
        self.components = None
        self.popt = None
        self.pcov = None
        self.converted_popt = None
        self.converted_pcov = None
        self.residuals = None
        self.sumsqr = None
        self.R2 = None

    def convert(self) -> Self:
        """
        Converts popt, containing model coefficients to proper units Uses metadata to
        add units proper units to various metrics.

        Returns:
            self: with converted data

        """
        self.converted_popt = copy.deepcopy(self.popt)
        self.converted_pcov = copy.deepcopy(self.pcov)
        # A one component models lack a population variable
        # Multiple components follow the same coefficient format
        # Pop_n, Pop_n+1..., Pop_m, Constant_n, Constant_n+1..., Constant_m
        if self.kinetic.name == "BSL":
            if self.components == 1:
                self.converted_popt[0] = self.popt[0] * self.microscope.framestep_size
                self.converted_pcov[0] = self.pcov[0] * self.microscope.framestep_size
            else:
                self.converted_popt[self.components :] = (
                    self.popt[self.components :] * self.microscope.framestep_size
                )
                self.converted_pcov[self.components :] = (
                    self.pcov[self.components :] * self.microscope.framestep_size
                )
        elif self.kinetic.name == "RayDifCoef":
            if self.components == 1:
                self.converted_popt[0] = (
                    1e8 * self.popt[1] ** 2 / (2 * self.microscope.framestep_size)
                )
                self.converted_pcov[0] = (
                    1e8 * self.pcov[1] ** 2 / (2 * self.microscope.framestep_size)
                )
            else:
                self.converted_popt[self.components :] = (
                    1e8
                    * self.popt[self.components :] ** 2
                    / (2 * self.microscope.framestep_size)
                )
                self.converted_pcov[self.components :] = (
                    1e8
                    * self.pcov[self.components :] ** 2
                    / (2 * self.microscope.framestep_size)
                )
        return self

    def dictify(self) -> Union[np.nan, Dict[str, any]]:
        """
        Format curve fitting data as a dictionary, necessary for appending data to
        dataframe.

        Returns:
            Union[np.nan, Dict[str, any]]: if model fitting fails, returns np.nan; if successful,
            return dictionary of model information

        """
        # Sometimes model fitting will fail (data cannot be described)
        if np.nan in self.popt:
            logger.warning(
                f"'{self.model_name}' model fitting failed, no data for update."
            )
            return np.nan
        export_dict = {}
        prefix = f"{self.model_name} : {self.kinetic.name}"
        if self.components == 1:
            new_dict = {
                f"{prefix} ({self.kinetic.unit})": self.converted_popt[0],
                f"{prefix} Cov": self.converted_pcov[0],
            }
            export_dict.update(new_dict)
        elif self.components > 1:
            total_popt = sum(self.popt[: self.components])
            for i in range(self.components):
                position = ordinal(i + 1)
                new_dict = {
                    f"{prefix} {position} Pop %": 100 * self.popt[i] / total_popt,
                    f"{prefix} {position} Pop Cov": self.pcov[i],
                    f"{prefix} {position} Metric ({self.kinetic.unit})": (
                        self.converted_popt[i + self.components]
                    ),
                    f"{prefix} {position} Metric Cov": self.converted_pcov[
                        i + self.components
                    ],
                }
                export_dict.update(new_dict)
        remainder_dict = {
            f"{prefix} SumSqr": self.sumsqr,
            f"{prefix} R\u00b2": self.R2,
        }
        export_dict.update(remainder_dict)
        return export_dict

    def __str__(self):
        """
        When calling class object as a string, return a string readout of the object's
        attributes.
        """
        return str(self.__dict__)


class BaseModelBuilder(metaclass=ABCMeta):
    """
    Base model builder.
    """

    def __init__(self) -> None:
        """
        Creates blank model attribute.
        """
        self.model = None

    def generate_model(self) -> None:
        """
        Establishes blank model object.
        """
        self.model = Model()

    @abstractmethod
    def set_equation(self) -> None:
        """
        Overridden by concrete classes.

        Raises:
            NotImplementedError: method is not overridden

        """
        raise NotImplementedError

    @abstractmethod
    def set_guess(self) -> None:
        """
        Overridden by concrete classes.

        Raises:
            NotImplementedError: method is not overridden

        """
        raise NotImplementedError

    @abstractmethod
    def add_attributes(self) -> None:
        """
        Overridden by concrete classes.

        Raises:
            NotImplementedError: method is not overridden

        """
        raise NotImplementedError

    @abstractmethod
    def calculate_parameters(self) -> None:
        """
        Overridden by concrete classes.

        Raises:
            NotImplementedError: method is not overridden

        """
        raise NotImplementedError


class Director:
    """
    Directs model building.
    """

    def __init__(self, builder: Type[BaseModelBuilder]) -> None:
        """
        Prep director for building.

        Args:
            builder (object): builds the model object

        """
        self._builder = builder

    def build_model(self) -> Self:
        """
        Calls functions to build model All these functions share methods across classes
        that allow for homogenous model object creation.

        Returns:
            self: director after creating object with specified builder

        """
        # For additional model-able components, class structure
        # must contain these functions
        self._builder.generate_model()
        self._builder.set_equation()
        self._builder.set_guess()
        self._builder.add_attributes()
        self._builder.calculate_parameters()
        return self

    def get_model(self) -> None:
        """
        Retrieves generated model.

        Returns:
            model object

        """
        return self._builder.model.convert()


class ExpDecay(BaseModelBuilder):
    """
    Exponential Decay model builder.

    Args:
        BaseModelBuilder (class): model builder base class

    """

    component_options = [1, 2, 3, "ExpLin"]  # ExpLin is currently unused

    def __init__(
        self,
        microscope: metadata.Microscope,
        movie: metadata.Movie,
        *,
        components: int or str,
        kinetic: kin.Kinetic,
        table: pd.DataFrame,
    ) -> None:
        """
        Initialize the exponential decay object.

        Args:
            microscope (metadata.Microscope): persistent microscope attributes
            movie (metadata.Movie): movie specific attributes
            components (int): number of model components
            kinetic (kin.Kinetic): kinetic for model fitting
            table (pd.DataFrame): formatted data for fitting

        """
        assert (
            components in ExpDecay.component_options
        ), "Not a valid component description"
        super().__init__()
        self.microscope = microscope
        self.movie = movie
        self._components = components
        self.kinetic = kinetic
        self._df = table
        self._equation = None
        self._limits = None
        self._guess = None

    def set_equation(self) -> None:
        """
        Finds equation for fitting based on components Equation selection is based on a
        string completion finder It is necessary to carefully name models for finding
        algorithm.

        Raises:
            TypeError: component value must be within class options

        """
        try:
            if isinstance(self._components, int):
                # i.e. 1, 2, 3... component models
                model_name = f"{num2words(self._components).capitalize()}Comp{self.__class__.__name__}"
                self._equation = getattr(Equations, model_name)
                NUM_LIMITS = len(signature(self._equation).parameters) - 1
                # Because time and population cannot be negative, lower limit is '0'
                self._limits = self._limits = (
                    [0.0 for i in range(NUM_LIMITS)],
                    [np.inf for i in range(NUM_LIMITS)],
                )

            elif isinstance(self._components, str):
                # i.e. Exponential and Linear decays
                model_name = f"{self._components}Comp{self.__class__.__name__}"
                self._equation = getattr(Equations, model_name)
                NUM_LIMITS = len(signature(self._equation).parameters) - 1
                self._limits = (
                    [0 for i in range(NUM_LIMITS)],
                    [np.inf for i in range(NUM_LIMITS)],
                )
            else:
                # Error will only raise if invalid model type is chosen
                # This is a good catching user-generated models
                raise TypeError
            self.model.equation = self._equation
        except TypeError:
            print("Component descriptor is not a valid type (int or str)")

    def set_guess(self) -> None:
        """
        Set model fitting origin point Guesses are based on previous data Biologic
        properties are good starting points for guesses.
        """
        # estimates based on e^(-t/tau)
        tau_estimation = e_estimation(self._df)
        if self._components == 1:
            self._guess = [tau_estimation]
        # Guesses are generated to match number of equation dependent variables
        elif self._components > 1:
            major_population = (1 / self._components) * 1.5
            minor_population = (1 - major_population) / (self._components - 1)
            self._guess = (
                [major_population]
                + list(itertools.repeat(minor_population, self._components - 1))
                + list(itertools.repeat(tau_estimation, self._components))
            )
        else:
            self._guess = [tau_estimation, 0.3]

    def add_attributes(self) -> None:
        """
        Add basic attributes.
        """
        self.model.microscope = self.microscope
        self.model.movie = self.movie
        self.model.kinetic = self.kinetic
        self.model.model_name = f"{self._components}-Comp {self.__class__.__name__}"
        self.model.components = self._components

    def calculate_parameters(self) -> None:
        """
        Call the curve fitting function method.
        """
        (
            self.model.popt,
            self.model.pcov,
            self.model.residuals,
            self.model.R2,
        ) = FitFunction.curve(self._df, self._equation, self._guess, self._limits)
        self.model.sumsqr = np.sum(self.model.residuals**2)


class RayDiff(BaseModelBuilder):
    """
    Rayleigh Distribution / Diffusion.

    Args:
        BaseModelBuilder (class): inherited model values

    """

    component_options = [1, 2, 3]

    def __init__(
        self,
        microscope: metadata.Microscope,
        movie: metadata.Movie,
        *,
        components: int,
        kinetic: kin.Kinetic,
        table: pd.DataFrame,
    ) -> None:
        """
        Initialize Rayleigh object.

        Args:
            microscope (metadata.Microscope): persistent metadata
            movie (metadata.Movie): movie object
            components (int): number of components
            kinetic (kin.Kinetic): kinetic class object
            table (pd.DataFrame): formatted data for fitting

        """
        assert (
            components in RayDiff.component_options
        ), "Not a valid component description"
        self.microscope = microscope
        self.movie = movie
        self._components = components
        self.kinetic = kinetic
        self._df = table
        self._equation = None
        self._limits = None
        self._guess = None

    def set_equation(self) -> None:
        """
        Finds equation for fitting based on components Equation selection is based on a
        string completion finder It is necessary to carefully name models for finding
        algorithm.
        """
        model_name = (
            f"{num2words(self._components).capitalize()}Comp{self.__class__.__name__}"
        )
        self._equation = getattr(Equations, model_name)
        NUM_LIMITS = len(signature(self._equation).parameters) - 1
        self._limits = (
            [0.0 for i in range(NUM_LIMITS)],
            [np.inf for i in range(NUM_LIMITS)],
        )
        self.model.equation = self._equation

    def set_guess(self) -> None:
        """
        Set model fitting origin point Guesses are based on previous data Biologic
        properties are good starting points for guesses.
        """
        try:
            if self._components == 1:
                self._guess = [1e-4, 1e-5]
            elif self._components == 2:
                self._guess = [7e-3, 2.7e-3, 2.87e-5, 9.33e-6]
            elif self._components > 2:
                self._guess = [7e-3 / 2 for i in range(self._components * 2)]
            else:
                raise ValueError
        except ValueError:
            logger.warning(f"No available guesses for {self._components}-Model")

    def add_attributes(self) -> None:
        """
        Add basic attributes.
        """
        self.model.microscope = self.microscope
        self.model.movie = self.movie
        self.model.kinetic = self.kinetic
        self.model.model_name = f"{self._components}-Comp {self.__class__.__name__}"
        self.model.components = self._components

    def calculate_parameters(self) -> None:
        """
        Call curve fitting model.
        """
        (
            self.model.popt,
            self.model.pcov,
            self.model.residuals,
            self.model.R2,
        ) = FitFunction.curve(self._df, self._equation, self._guess, self._limits)
        self.model.sumsqr = np.sum(self.model.residuals**2)


class Equations:
    """
    Holder for all fitting equations Additional equations can be added as long as they
    match the formatting of the currently present equation.

    Focus on the order of population and time related variables.

    """

    @staticmethod
    def OneCompExpDecay(t: float, tau: float) -> np.ndarray:
        """
        One Component Exponential Decay.

        Args:
            t (float): independent time variable
            tau (float): dependent time constant

        Returns:
            equation: full equation for model fitting

        """
        return np.exp(-t / tau)

    @staticmethod
    def TwoCompExpDecay(
        t: float, a: float, b: float, tau1: float, tau2: float
    ) -> np.ndarray:
        """
        Two Component Exponential Decay.

        Args:
            t (float): independent time variable
            a (float): population constant
            b (float): population constant
            tau1 (float): time constant
            tau2 (float): time constant

        Returns:
            equation: full equation for model fitting

        """
        return (a * np.exp(-t / tau1)) + (b * np.exp(-t / tau2))

    @staticmethod
    def ThreeCompExpDecay(
        t: float, a: float, b: float, c: float, tau1: float, tau2: float, tau3: float
    ) -> np.ndarray:
        """
        Three Component Exponential Decay.

        Args:
            t (float): independent time variable
            a (float): population constant
            b (float): population constant
            c (float): population constant
            tau1 (float): time constant
            tau2 (float): time constant
            tau3 (float): time constant

        Returns:
            equation: full equation for model fitting

        """
        return (
            (a * np.exp(-t / tau1)) + (b * np.exp(-t / tau2)) + (c * np.exp(-t / tau3))
        )

    @staticmethod
    def OneCompRayDiff(t: float, a: float, sig: float) -> np.ndarray:
        """
        One Component Rayleigh Distrubution.

        Args:
            t (float): independent time variable
            a (float): scale constant
            sig (float): time constant

        Returns:
            equation: full equation for model fitting

        """
        return a * ((t * np.exp(-(t**2) / (2 * sig**2))) / sig**2)

    @staticmethod
    def TwoCompRayDiff(
        t: float, a: float, b: float, sig1: float, sig2: float
    ) -> np.ndarray:
        """
        Two Component Rayleigh Distribution.

        Args:
            t (float): independent time variable
            a (float): population constant
            b (float): population constant
            sig1 (float): time constant
            sig2 (float): time constant

        Returns:
            equation: full equation for model fitting

        """
        return (a * (t * np.exp(-(t**2) / (2 * sig1**2))) / sig1**2) + (
            b * (t * np.exp(-(t**2) / (2 * sig2**2))) / sig2**2
        )

    @staticmethod
    def ThreeCompRayDiff(
        t: float, a: float, b: float, c: float, sig1: float, sig2: float, sig3: float
    ) -> np.ndarray:
        """
        Three Component Rayleigh Distribution.

        Args:
            t (float): independent time variable
            a (float): population constant
            b (float): population constant
            c (float): population constant
            sig1 (float): time constant
            sig2 (float): time constant
            sig3 (float): time constant

        Returns:
            equation: full equation for model fitting

        """
        return (
            (a * (t * np.exp(-(t**2) / (2 * sig1**2))) / sig1**2)
            + (b * (t * np.exp(-(t**2) / (2 * sig2**2))) / sig2**2)
            + (c * (t * np.exp(-(t**2) / (2 * sig3**2))) / sig3**2)
        )

    @staticmethod
    def ExpLinCompExpDecay(t: float, tau: float, slope: float) -> np.ndarray:
        """
        Exponential and Linear Decay.

        Args:
            t (float): indpenedent time variable
            tau (float): exponential decay time constant
            slope (float): linear decay time constant

        Returns:
            _type_: _description_

        """
        return np.exp(-t / tau) + slope * t


class FitFunction:
    """
    Fitting methods.
    """

    @staticmethod
    def curve(
        df: pd.DataFrame,
        equation: np.ndarray,
        p0: list[float],
        limits: tuple = None,
    ) -> tuple:
        """
        Fits curved data.

        Args:
            df (pd.DataFrame): formatted data for model fitting
            equation (np.ndarray): model equation
            p0 (list): initial guesses for model fitting
            limits (list, optional): min and max constant values. Defaults to None.

        Returns:
            tuple: constants, variance, R-squared, and sum of square residuals

        """
        try:
            x_data = df.iloc[:, 0].values.astype(float)
            y_data = df.iloc[:, 1].values.astype(float)
            # Putting limits as None will cause errors
            if limits:
                popt, pcov = curve_fit(equation, x_data, y_data, p0=p0, bounds=limits)
            else:
                popt, pcov = curve_fit(equation, x_data, y_data, p0=p0)
            pcov = np.sqrt(np.diag(pcov))
            RES = y_data - equation(x_data, *popt)
            SUM_SQR_RES = np.sum(RES**2)
            SUM_SQR_TOT = np.sum((y_data - np.mean(y_data)) ** 2)
            R2 = 1 - (SUM_SQR_RES / SUM_SQR_TOT)
            return popt, pcov, RES, R2
        except RuntimeError:
            return np.nan, np.nan, np.nan, np.nan

    @staticmethod
    def linear(df: pd.DataFrame) -> tuple:
        """
        Linear Regression.

        Args:
            df (pd.DataFrame): two column dataframe

        Returns:
            tuple: line slope, y-axis intercept, and R-squared

        """
        assert df.shape[1] == 2, "Dataframe must contain only two columns"
        x_data = df.iloc[:, 0]
        y_data = df.iloc[:, 1]
        slope, intercept, r2 = stats.linregress(x_data, y_data)
        return slope, intercept, r2
