"""
SmToolS (s)ingle (m)olecule (Tool) by (S)tefan.
"""

import argparse
import logging
import operator
import os
import sys
from datetime import datetime
from functools import reduce

import numpy as np
import pandas as pd

script_path = os.path.realpath(__file__)
tool_path = os.path.realpath(os.path.join(script_path, "..", "..", ".."))
sys.path.insert(0, tool_path)

import smToolS.metadata as metadata
from smToolS.sm_helpers import constants as cons
from smToolS.sm_helpers import parsers
from smToolS.sm_helpers.handlers import analyze_handler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_parser():
    arg_parser = argparse.ArgumentParser(
        parents=[
            parsers.ImportParser.get_parser(),
            parsers.ScriptParser.get_parser(),
            parsers.MicroscopeParser.get_parser(),
            parsers.ExportParser.get_parser(),
        ]
    )
    return arg_parser


def trio_lyze(microscope: metadata.Microscope, df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate average brightness, length, and diffusivity for each trajectory.

    Args:
        microscope (metadata.Microscope): microscope parameters (ex. frame rate)
        df (pd.DataFrame): trajectory data

    Returns:
        pd.DataFrame: updated trajectory data with trio

    """
    logger.info("Beginning Pre-Processing.")
    if cons.BRIGHTNESS in df.columns:
        df[cons.AVERAGE_BRIGHTNESS] = df.groupby(cons.TRAJECTORY)[
            cons.BRIGHTNESS
        ].transform(np.mean)
    df[cons.LENGTH_W_UNITS] = df.groupby(cons.TRAJECTORY)[cons.TRAJECTORY].transform(
        "size"
    )
    data = df.groupby(cons.TRAJECTORY)[[cons.Coordinates.X, cons.Coordinates.Y]].apply(
        microscope.calc_one_step_MSD
    )
    data = pd.DataFrame(data.to_list(), columns=[cons.SDS, cons.MSD_W_UNITS])
    data.index += 1
    data.reset_index(inplace=True)
    data = data.rename(columns={"index": cons.TRAJECTORY})
    df[cons.SDS] = reduce(operator.add, data[cons.SDS])
    df = df.merge(data[[cons.TRAJECTORY, cons.MSD_W_UNITS]])
    logger.info("Completed Pre-Processing.")
    return df


def analyze_movie(
    file: tuple,
    script: metadata.Script,
    microscope: metadata.Microscope,
    save_location: str,
) -> pd.DataFrame:
    """
    Main processing pipeline Reads in movies and calculates desired metrics based on
    trajectory data.

    Args:
        file (tuple): filename and filedata tuple
        script: script object
        microscope: microscope object

    Returns:
        pd.DataFrame: export data for specific movie

    """
    file_info = file[0]
    logger.info(f"Beginning Movie Analysis: '{file_info}'.")
    start_time = datetime.now()
    movie_path, trajectories = file
    movie = metadata.Movie(script, movie_path, trajectories)
    movie.update_trajectory_df(new_df=trio_lyze(microscope, movie.data_df))
    handler = analyze_handler.AnalyzeHandler(script, microscope, movie)
    handler.implement_cutoffs().construct_kinetics().generate_models()
    handler.display_and_or_save(save_location)
    end_time = datetime.now()
    logger.info(f"Completed Movie Analysis: '{file_info}'.")
    # only log time if there are no pauses to look at figures
    if (
        script.brightness_method != cons.CutoffMethods.MANUAL
        and script.display is False
    ):
        logger.info(f"Elapsed Time: '{(end_time - start_time).total_seconds()}' sec.")
    return pd.DataFrame(handler.movie.export_dict, index=[0])


def main(parser_args: argparse.Namespace):
    logger.info("Beginning Analysis.")
    parsers.ImportParser.validate(parser_args)
    save_location = parsers.ExportParser.set_save_location(parser_args)
    script = metadata.Script(
        parser_args.filetype,
        parser_args.file,
        parser_args.directory,
        parser_args.display,
        parser_args.save_images,
        parser_args.cutoffs,
        parser_args.brightness_method,
        parser_args.min_length,
        # parser_args.diffusion_method,
        parser_args.brightness_cutoffs,
        parser_args.diffusion_cutoffs,
    )
    microscope = metadata.Microscope(parser_args.pixel_size, parser_args.framestep_size)
    file_info = metadata.FileReader(
        script.filetype, script.file_list, parser_args.framestep_size
    )

    all_dfs = []
    for file in file_info.pre_processed_files:
        df = analyze_movie(file, script, microscope, save_location)
        all_dfs += [df]
    dfs = pd.concat(all_dfs)
    metadata.Export(dfs, save_location, parser_args.export_as)
    logger.info("Analysis Complete, ending program.")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
