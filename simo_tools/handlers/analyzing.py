# import logging
# from typing import Self, Type

# from simo_tools import movie as mov

# # from simo_tools.analysis import kinetics as kin

# logging.basicConfig()
# logger = logging.getLogger(__name__)
# logger.setLevel(logging.INFO)


# class Analyzer:
#     """
#     Handles execution of cutoffs, kinetic creation, modeling, and so on.
#     """

#     # ALL_KINETICS: list[type[kin.Kinetic]] = [kin.BSL, kin.MSD, kin.RayD]
#     NUM_MODELS: float = 3  # we only have up max three component models

#     def __init__(
#         self,
#         movie: mov.Movie,
#         pixel_size: float,
#         fps: float,
#     ):
#         self._movie = movie
#         self._pixel_size = pixel_size
#         self._fps = fps

#     def construct_kinetics(self) -> Self:
#         director = kin.Director
#         for kinetic in self.ALL_KINETICS:
#             director(kinetic).construct_kinetic().get_kinetic()
#         self._bsl = (
#             kin.Director(kin.BSL(self._script, self._movie.data_df))
#             .construct_kinetic()
#             .get_kinetic()
#         )
#         self._msd = (
#             kin.Director(kin.MSD(self._microscope, self._movie.data_df))
#             .construct_kinetic()
#             .get_kinetic()
#         )
#         self._movie.add_export_data(kin.MSD.model(self._msd.table))
#         self._rayd = (
#             kin.Director(
#                 kin.RayD(self._microscope, self._movie.data_df[cons.SDS].dropna())
#             )
#             .construct_kinetic()
#             .get_kinetic()
#         )
#         return self

#     def generate_models(self) -> Self:
#         for i in range(1, self.NUM_MODELS + 1):
#             decay_model = (
#                 cf.Director(
#                     cf.ExpDecay(
#                         self._microscope,
#                         self._movie,
#                         components=i,
#                         kinetic=self._bsl,
#                         table=self._bsl.table,
#                     )
#                 )
#                 .build_model()
#                 .get_model()
#             )
#             self._models += [decay_model]
#             self._movie.add_export_data(decay_model.dictify())

#             diffusion_model = (
#                 cf.Director(
#                     cf.RayDiff(
#                         self._microscope,
#                         self._movie,
#                         components=i,
#                         kinetic=self._rayd,
#                         table=self._rayd.table,
#                     )
#                 )
#                 .build_model()
#                 .get_model()
#             )
#             self._models += [diffusion_model]
#             self._movie.add_export_data(diffusion_model.dictify())
#         return self

#     def display_and_or_save(self, save_location: str):
#         if self._script.display or self._script.save_images:
#             for model in self._models:
#                 figure = di.ScatteredLine(model)
#                 figure.plot(
#                     self._script.display, self._script.save_images, save_location
#                 )
