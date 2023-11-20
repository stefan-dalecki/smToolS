import os

import pandas as pd
import pytest

from simo_tools import metadata as meta


@pytest.fixture
def trajectories_df(test_data_loc: str):
    """
    Dataframe of Trajectories.
    """
    return pd.read_csv(os.path.join(test_data_loc, "test_trajectories.csv"))


@pytest.fixture
def trajectories_obj(trajectories_df: pd.DataFrame) -> meta.Trajectories:
    """
    Trajectories as a `metadata.Trajectories` object.
    """
    return meta.Trajectories.from_df(trajectories_df)
