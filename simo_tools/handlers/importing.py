import os
import pandas as pd

from simo_tools.helpers import constants as cons


class FileReader:
    PROCESS_METHOD_MAP: dict[cons.ReadFileTypes, callable] = {
        ReadFileTypes.CSV: _process_csv
    }
    """
    Loads an optionally (for nd2 files) tracks trajectories.
    """

    def __init__(self, path: str):
        if os.path.isdir(path):
            self.filepaths = set(os.listdir(path))
        elif os.path.isfile(path):
            self.filepaths = set(path)
        else:
            raise TypeError(f"Given path: `{path}` is not a file or directory.")

    def pre_process(self, filetype: str):
        """
        Processes subset of filepaths that match filetype, using specific filetype
        method.

        Args:
            filetype (str): ex: "csv"

        """
        try:
            process_method = cons.ReadFileTypes[filetype]
        except KeyError as exc:
            raise ValueError(
                f"Given filetype: `{filetype}` must be one of"
                f" {', '.join(cons.ReadFileTypes.set_of_options())}."
            ) from exc

    def _process_csv(self):
        for file in self.filepaths:
            df = pd.read_csv(file, index_col=[0])
            df.rename(
                columns={
                    cons.PARTICLE: cons.TRAJECTORY,
                    cons.M2: cons.Cutoffs.BRIGHTNESS,
                },
                inplace=True,
            )
            track_df = self._sanitize_column_names(track_df)
            self.pre_processed_files += [(file, track_df)]

class Trajectory:
    pass

class TrajectoryTable:
    def __init__(self, df: pd.DataFrame):
