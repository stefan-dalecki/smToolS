import os

import pandas as pd
import pytest


@pytest.fixture
def raw_trajectories_csv_path(test_data_loc: str):
    """
    Raw file with unformatted columns.
    """
    return os.path.join(test_data_loc, "test__process_csv.csv")


@pytest.fixture
def raw_trajectories_csv_df(raw_trajectories_csv_path: str):
    """
    Raw file with unformatted columns.
    """
    df = pd.read_csv(raw_trajectories_csv_path, index_col=[0])
    return df
