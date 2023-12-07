import os
from dataclasses import dataclass
from functools import reduce
from typing import Optional, Self, cast

from simo_tools import constants as cons
from simo_tools import movie as mov
from simo_tools import trajectory as traj
from simo_tools.analysis import cutoffs as cuts
from simo_tools.handlers import importing

__all__ = [
    "FileData",
]
CutoffTypehint = Optional[dict[str, float | cons.CutoffMethods | None]]


def _validate_filetype(filetype: str) -> cons.ReadFileTypes:
    """
    Esnures filetype is one of `cons.ReadFileTypes`
    """
    try:
        read_filetype = cast(cons.ReadFileTypes, cons.ReadFileTypes[filetype])
    except KeyError as exc:
        raise ValueError(
            f"Detected filetype `{filetype}` is not one of:"
            f" {' ,'.join(cons.ReadFileTypes.set_of_options())}."
        ) from exc

    return read_filetype


@dataclass
class FileData:
    """
    Loads an optionally (for nd2 files) tracks trajectories.
    """

    filetype: cons.ReadFileTypes
    movies: list[mov.Movie]

    # changed via `set_units`
    pixel_size: Optional[float] = None
    fps: Optional[float] = None

    # set during `apply_cutoffs`
    cutoffs: Optional[list[type[cuts.Cutoff]]] = None

    @classmethod
    def from_path(
        cls, path: str, filetype: Optional[str | cons.ReadFileTypes] = None
    ) -> Self:
        """
        Generates class from file or directory path.
        """
        if os.path.isdir(path):
            assert filetype, "Filetype must be given if searching through directory."
            filetype = _validate_filetype(importing.get_filetype(path))
            filepaths = cls._detect_matching_files(path, filetype)
        elif os.path.isfile(path):
            filepaths = {path}
            filetype = _validate_filetype(importing.get_filetype(path))
        else:
            raise TypeError(f"Given path: `{path}` is not a file or directory.")

        movies = [
            mov.Movie.from_path(filepath, self.pixel_size, self.fps)
            for filepath in filepaths
        ]

        obj = cls(
            filetype=cast(cons.ReadFileTypes, filetype),
            movies=movies,
        )
        return obj

    @staticmethod
    def _detect_matching_files(path: str, filetype: str) -> set[str]:
        filepaths = set()
        for *_, files in os.walk(path):
            for name in files:
                if name.endswith(filetype):
                    filepaths.update(name)
        return filepaths

    # @property
    # def tables(self) -> list[pd.DataFrame | None]:
    #     """
    #     Imported tables.
    #     """
    #     return [imported_file.table for imported_file in self.imported_files]

    def set_units(self, pixel_size: float, fps: float) -> Self:
        """
        Establishes unit measurements that are shared across all movies.

        Args:
            pixel_size (float): µ²/sec
            framestep_size (float): frames/sec

        """
        self.pixel_size = pixel_size
        self.fps = fps
        return self

    def apply_cutoffs(self, cutoffs: list[type[cuts.Cutoff]]) -> Self:
        """
        Sets cutoff attrs based on given dictionary.
        """
        # this could probably be mapped
        self.cutoffs = cutoffs
        for movie in self.movies:
            all_valid_trajs = []
            for cutoff in cutoffs:
                all_valid_trajs += [cutoff.threshold(movie.trajectories)]

            valid_trajs = reduce(traj.get_shared_trajectories, all_valid_trajs)
            movie.thresholded_trajectories = valid_trajs
        return self

    def analyze(self):
        assert (
            self.pixel_size and self.fps
        ), "Both `pixel_size` and `fps` must be set prior to analyzing movies."
        for movie in self.movies:
            movie.analyze(pixel_size=self.pixel_size, fps=self.fps)
