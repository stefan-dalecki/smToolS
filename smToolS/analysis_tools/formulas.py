"""Various function shortcuts"""
import numpy as np
import pandas as pd

import smToolS.sm_helpers.constants as cons

# def all_steps(microscope: Microscope, df: pd.DataFrame) -> defaultdict[list]:
#     """Calculate MSD for 8 step lengths
#
#     Args:
#         microscope (Microscope): microscope parameters
#         df (pd.DataFrame): trajectory data
#
#     Returns:
#         defaultdict[list]: squared distance for each step length
#     """
#     df = df.reset_index(drop=True)
#     x_col, y_col = df[cons.Coordinates.X], df[cons.Coordinates.Y]
#     all_steps = defaultdict(list)
#     if len(df) - 3 > 8:
#         max_step_len = 8
#     else:
#         max_step_len = len(df) - 3
#     for step_len in range(1, max_step_len + 1):
#         for step_num in range(0, len(df) - step_len - 1):
#             x1, y1 = x_col[step_num], y_col[step_num]
#             x2, y2 = x_col[step_num + step_len], y_col[step_num + step_len]
#             distance = (
#                     calc_distance(x1, y1, x2, y2) ** 2
#                     * microscope.pixel_size ** 2
#                     / (4 * microscope.framestep_size)
#             )
#             all_steps[step_len].append(distance)
#     return all_steps


def calc_distance(x1: float, y1: float, x2: float, y2: float) -> float:
    """Calculate distance between two coordinates using pythagorean theorem

    Args:
        x1 (float): x-value
        y1 (float): y-value
        x2 (float): x2-value
        y2 (float): y2-value

    Returns:
        float: distance between two coordinates
    """
    return (((x2 - x1) ** 2) + ((y2 - y1) ** 2)) ** (1 / 2)


def depth_of_field(lam=532, incidence=69, n2=1.33, n1=1.515) -> float:
    """smTIRF depth of field

    Args:
        lam (int, optional): laser wavelength. Defaults to 532.
        incidence (int, optional): incidence. Defaults to 69.
        n2 (float, optional): refractive index 2. Defaults to 1.33.
        n1 (float, optional): refractive index. Defaults to 1.515.

    Returns:
        float: microscope depth of field
    """

    # Though not ever called, this equation can be used to calculate a constant in
    # the luminescence function below
    theta = incidence * (180 / np.pi)
    depth_of_field = (lam / (4 * np.pi)) * (n1**2 * (np.sin(theta)) ** 2 - n2**2) ** (-1 / 2)
    return depth_of_field


def f_modeltest(ss1: float, ss2: float, degfre1: int, degfre2: int) -> float:
    """Generate f-statistic comparing two models

    Args:
        ss1 (float): sums squared
        ss2 (flaot): sums square 2
        degfre1 (int): degrees of freedom
        degfre2 (int): degrees of freedom 2

    Returns:
        float: f-statistic
    """
    # F-model tests are unsuccessful with models that contain a large number of
    # measured values (i.e. a range of 100 x-values). They are successful when a
    # model is fit to few data points (i.e. less than 20)
    if not degfre1 > degfre2:
        raise ValueError(
            f"First degree of freedom '{degfre1}' must be greater than second '{degfre2}'."
        )
    f_stat = ((ss1 - ss2)(degfre1 - degfre2)) / (ss2 / degfre2)
    return f_stat


def input_bool(input_string: str) -> bool:
    """User input that results in true or false

    Args:
        input_string (str): question for user

    Returns:
        bool: binary decision
    """

    while True:
        ans = input(f"\n{input_string} (y/n): ")
        if ans == "y":
            ans = True
            break
        elif ans == "n":
            ans = False
            break
        else:
            print("Not an available option\n")
    return ans


def luminescence(I0: float, d: float, z: float) -> float:
    """Intensity at distances off of coverslip

    Args:
        I0 (float): initial intensity
        d (float): depth of field
        z (float): distance from coverslip

    Returns:
        float: intensity at z-distance
    """

    Iz = I0 ** (-z / d)
    return Iz


def trajectory_count(df: pd.DataFrame) -> int:
    """Counts numbers of trajectories remaining

    Args:
        df (pd.DataFrame): trajectory data

    Returns:
        int: number of trajectories
    """
    return df[cons.TRAJECTORY].nunique()
