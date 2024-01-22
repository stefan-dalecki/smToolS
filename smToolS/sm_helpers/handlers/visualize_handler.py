import logging
import operator
from functools import reduce
from typing import Dict, Optional, Self

import numpy as np
import pandas as pd

from smToolS import metadata
from smToolS.analysis_tools import display as di
from smToolS.sm_helpers import constants as cons

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class VisualizeHandler:
    def __init__(
        self, df: pd.DataFrame, script: metadata.Script, microscope: metadata.Microscope
    ):
        self.df = df
        self._script = script
        self._microscope = microscope

    def trio_lyze(self) -> Self:
        """Calculate average brightness, length, and diffusivity for each trajectory"""
        logger.info("Beginning Pre-Processing")
        self.df[cons.AVERAGE_BRIGHTNESS] = self.df.groupby(cons.TRAJECTORY)[
            cons.BRIGHTNESS
        ].transform(np.mean)
        self.df[cons.LENGTH_W_UNITS] = self.df.groupby(cons.TRAJECTORY)[
            cons.TRAJECTORY
        ].transform("size")
        data = self.df.groupby(cons.TRAJECTORY)[
            [cons.Coordinates.X, cons.Coordinates.Y]
        ].apply(self._microscope.calc_one_step_MSD)
        data = pd.DataFrame(data.to_list(), columns=[cons.SDS, cons.MSD_W_UNITS])
        data.index += 1
        data.reset_index(inplace=True)
        data = data.rename(columns={"index": cons.TRAJECTORY})
        self.df[cons.SDS] = reduce(operator.add, data[cons.SDS])
        self.df = self.df.merge(data[[cons.TRAJECTORY, cons.MSD_W_UNITS]])
        logger.info("Completed Pre-Processing")
        return self

    def identify_fractions(
        self,
        sep: Optional[str] = "\u00b7",
        keepers: Optional[str] = "Valid",
        *,
        criteria: Optional[Dict[str, tuple]] = None,
    ) -> Self:
        logger.info("Beginning to identify fractions.")
        if criteria is None:
            criteria = {
                "dim": (cons.AVERAGE_BRIGHTNESS, "<", 3.1),
                "bright": (cons.AVERAGE_BRIGHTNESS, ">", 3.8),
                "short": (cons.LENGTH_W_UNITS, "<", 10),
                "slow": (cons.MSD_W_UNITS, "<", 0.3),
                "fast": (cons.MSD_W_UNITS, ">", 3.5),
            }
        ops = {
            "<": operator.lt,
            "=<": operator.le,
            ">": operator.gt,
            "=>": operator.ge,
            "=": operator.eq,
        }
        ID = "ID"
        for key, val in criteria.items():
            col, op, num = val[0], val[1], val[2]
            self.df.loc[ops[op](self.df[col], num), key] = f"{key} {sep} "
        self.df = self.df.fillna("")
        self.df[ID] = self.df.iloc[:, -len(criteria) :].sum(axis=1)
        # Removes ending from last addition
        self.df[ID] = self.df[ID].str.rstrip(f" {sep} ")
        # Trajectories with no tag are set to the keepers variable_
        self.df.loc[self.df[ID] == "", ID] = keepers
        logger.info("Completed identifying fractions.")
        return self

    def display_and_or_save(self, save_location: str, save_name: str):
        figures = di.VisualizationPlots(
            self.df,
            self._script.display,
            self._script.save_images,
            save_location,
            save_name,
        )
        figures.two_dimensional_histogram()
        figures.three_dimensional_scatter()
