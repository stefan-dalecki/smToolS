import logging
from typing import Self

from smToolS import metadata
from smToolS.analysis_tools import curvefitting as cf
from smToolS.analysis_tools import cutoffs as cut
from smToolS.analysis_tools import display as di
from smToolS.analysis_tools import formulas as fo
from smToolS.analysis_tools import kinetics as kin
from smToolS.sm_helpers import constants as cons

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


class Analyzer:
    """
    Handles execution of cutoffs, kinetic creation, modeling, and so on.
    """

    NUM_MODELS = 3  # we only have up max three component models

    def __init__(
        self,
        script: metadata.Script,
        microscope: metadata.Microscope,
        movie: metadata.Movie,
    ):
        self._script = script
        self._microscope = microscope
        self._movie = movie
        self._bsl = None  # bound state lifetime
        self._msd = None  # mean squared displacement
        self._rayd = None  # rayleigh diffusion
        self._models = []

    @property
    def movie(self) -> metadata.Movie:
        """
        Returns the movie metadata object in its current state.

        Returns:
            metadata.Movie: movie object

        """
        return self._movie

    def implement_cutoffs(self) -> Self:
        logger.info(f"Beginning ---{self._script.cutoffs}--- Cutoffs")

        for cutoff in self._script.cutoffs:
            cutoff_method = getattr(self, f"_apply_{cutoff}")
            cutoff_method()

        self._movie.add_export_data({
            cons.TRAJECTORY_W_UNITS: fo.trajectory_count(self._movie.data_df)
        })
        logger.info(f"Completed ---{self._script.cutoffs}--- Cutoffs.")
        return self

    def _apply_brightness(self):
        if (
            self._script.brightness_method == "none"
            or self._script.filetype == cons.FileTypes.XML
        ):
            return  # no brightness cutoffs are needed
        elif self._script.brightness_method == cons.CutoffMethods.CLUSTERING:
            cluster = cut.Clustering(self._movie.data_df)
            cluster.scale_features().estimate_clusters().display().cluster()
            self._script.min_length = cluster.min_length
            self._movie.update_trajectory_df(new_df=cluster.cutoff_df)
        else:
            brightness = cut.Brightness(
                self._script.min_length,
                self._movie.data_df,
                self._script.brightness_method,
                self._script.brightness_cutoffs,
            )
            self._movie.update_trajectory_df(new_df=brightness.cutoff_df)

    def _apply_length(self):
        minimum_length = cut.Length(
            self._script.min_length, self._movie.data_df, method="minimum"
        )
        self._movie.update_trajectory_df(new_df=minimum_length.cutoff_df)

    def _apply_diffusion(self):
        diffusion = cut.Diffusion(self._movie.data_df, self._script.diffusion_cutoffs)
        self._movie.update_trajectory_df(new_df=diffusion.cutoff_df)

    def construct_kinetics(self) -> Self:
        logger.info("Beginning Constructing Kinetics.")
        self._bsl = (
            kin.Director(kin.BSL(self._script, self._movie.data_df))
            .construct_kinetic()
            .get_kinetic()
        )
        self._msd = (
            kin.Director(kin.MSD(self._microscope, self._movie.data_df))
            .construct_kinetic()
            .get_kinetic()
        )
        self._movie.add_export_data(kin.MSD.model(self._msd.table))
        self._rayd = (
            kin.Director(
                kin.RayD(self._microscope, self._movie.data_df[cons.SDS].dropna())
            )
            .construct_kinetic()
            .get_kinetic()
        )
        logger.info("Finished Constructing Kinetics.")
        return self

    def generate_models(self) -> Self:
        for i in range(1, self.NUM_MODELS + 1):
            decay_model = (
                cf.Director(
                    cf.ExpDecay(
                        self._microscope,
                        self._movie,
                        components=i,
                        kinetic=self._bsl,
                        table=self._bsl.table,
                    )
                )
                .build_model()
                .get_model()
            )
            self._models += [decay_model]
            self._movie.add_export_data(decay_model.dictify())

            diffusion_model = (
                cf.Director(
                    cf.RayDiff(
                        self._microscope,
                        self._movie,
                        components=i,
                        kinetic=self._rayd,
                        table=self._rayd.table,
                    )
                )
                .build_model()
                .get_model()
            )
            self._models += [diffusion_model]
            self._movie.add_export_data(diffusion_model.dictify())
        return self

    def display_and_or_save(self, save_location: str):
        if self._script.display or self._script.save_images:
            for model in self._models:
                figure = di.ScatteredLine(model)
                figure.plot(
                    self._script.display, self._script.save_images, save_location
                )
