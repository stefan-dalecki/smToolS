import os

import pandas as pd
import pytest

from simo_tools import trajectory as traj


@pytest.fixture
def trajectories_df(test_data_loc: str) -> pd.DataFrame:
    """
    Dataframe of Trajectories.
    """
    return pd.read_csv(os.path.join(test_data_loc, "test_trajectories.csv"))


@pytest.fixture
def trajectories_obj(trajectories_df: pd.DataFrame) -> traj.Trajectories:
    """
    Trajectories as a `metadata.Trajectories` object.
    """
    return traj.Trajectories.from_df(trajectories_df)
