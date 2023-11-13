import os

import pandas as pd
import pytest


@pytest.fixture
def trajectories_df(test_data_loc: str):
    df = pd.read_csv(os.path.join(test_data_loc, "test_trajectories.csv"))
    return df
