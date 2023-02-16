import argparse
import logging
import os
import sys

import pandas as pd

script_path = os.path.realpath(__file__)
tool_path = os.path.realpath(os.path.join(script_path, "..", "..", ".."))
sys.path.insert(0, tool_path)

from smToolS import metadata
from smToolS.sm_helpers import parsers
from smToolS.sm_helpers.handlers import visualize_handler

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def get_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        parents=[
            parsers.ImportParser.get_parser(),
            parsers.MicroscopeParser.get_parser(),
            parsers.ExportParser.get_parser(),
        ]
    )
    parser.add_argument(
        "--min-length",
        required=False,
        default=0,
        type=int,
        help="Minimum step length of trajectories in frames.",
    )
    parser.add_argument("--display", required=False, action="store_true")
    parser.add_argument("--save-images", required=False, action="store_true")
    return parser


def visualize_table(
    df: pd.DataFrame,
    script: metadata.Script,
    microscope: metadata.Microscope,
    save_location: str,
    save_name: str,
) -> None:
    handler = visualize_handler.VisualizeHandler(df, script, microscope)
    handler.trio_lyze().identify_fractions().display_and_or_save(save_location, save_name)


def main(parser_args: argparse.Namespace):
    if not parser_args.display or parser_args.save_images:
        raise ValueError(
            "This script is not worthwhile without specifying whether you want to display or save "
            "images."
        )
    logger.info("Beginning visualization.")
    parsers.ImportParser.validate(parser_args)
    save_location = parsers.ExportParser.set_save_location(parser_args)
    microscope = metadata.Microscope(parser_args.pixel_size, parser_args.framestep_size)
    script = metadata.Script(
        parser_args.filetype,
        parser_args.file,
        parser_args.directory,
        display=parser_args.display,
        save_images=parser_args.save_images,
        min_length=parser_args.min_length,
    )
    file_info = metadata.FileReader(script.filetype, script.file_list, parser_args.framestep_size)
    if parser_args.file:
        save_name = os.path.splitext(os.path.basename(parser_args.file))[0]
    else:
        save_name = None
    visualize_table(file_info.one_table, script, microscope, save_location, save_name)
    logger.info("Completed visualization.")


if __name__ == "__main__":
    args = get_parser().parse_args()
    main(args)
