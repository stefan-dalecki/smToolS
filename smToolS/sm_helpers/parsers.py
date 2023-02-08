import argparse
import os
from abc import ABC, abstractmethod

import smToolS.sm_helpers.constants as cons


def collect_as(coll_type):
    class CollectAs(argparse.Action):
        def __call__(self, arg_parser, namespace, values, options_string=None):
            setattr(namespace, self.dest, coll_type(values))

    return CollectAs


class BaseParser(ABC):
    @property
    @abstractmethod
    def get_parser(self):
        return NotImplemented

    @staticmethod
    def validate(self):
        return NotImplemented


class ImportParser(BaseParser):
    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--file",
            required=False,
            type=str,
            help="File that you wish to analyze.",
        )
        parser.add_argument(
            "--directory",
            required=False,
            type=str,
            help="Directory containing movies that you want to analyze.",
        )
        parser.add_argument(
            "--filetype",
            required=False,
            type=str,
            help="The file extension of your movies. Current support is for ImageJ/Particle Tracker Classic '.csv' files, "
            "'h5' for analyzed '.nd2' files within this program, and 'xml' files for TrackMate files. Only required if "
            "'file' is not given.",
        )
        return parser

    @staticmethod
    def validate(parser_args: argparse.Namespace):
        """
        Validation checks to make sure parser will not fail in pipeline.

        parser_args (argparse.Namespace): parser arguments

        Raises:
            RuntimeError: if parser does not meet specs
        """
        if not parser_args.file and not parser_args.directory:
            raise RuntimeError("File or file directory must be specified")
        if parser_args.directory and not parser_args.filetype:
            raise RuntimeError("If directory is given, filetype must be specified.")


class ScriptParser(BaseParser):
    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--cutoffs",
            required=False,
            default=cons.Cutoffs.list_of_options(),
            nargs="*",
            help="Select from the following options: brightness, length, diffusion. By default, all are used.",
        )
        parser.add_argument(
            "--brightness-method",
            required=False,
            default="auto",
            type=str,
            help="Method for implementing brightness cutoffs. See 'constants' file for options.",
        )
        parser.add_argument(
            "--brightness-cutoffs",
            required=False,
            nargs=2,
            type=float,
            action=collect_as(tuple),
            help="Low and high brightness cutoff values.",
        )
        parser.add_argument(
            "--min-length",
            required=False,
            default=0,
            type=int,
            help="Minimum step length of trajectories in frames.",
        )
        # parser.add_argument(
        #     "--diffusion-method",
        #     required=False,
        #     default="auto",
        #     type=str,
        #     help="Method for implementing diffusion cutoffs. See 'constants' file for options.",
        # )
        parser.add_argument(
            "--diffusion-cutoffs",
            required=False,
            nargs=2,
            type=float,
            action=collect_as(tuple),
            help="High and low diffusion cutoff values in um^2/sec.",
        )
        parser.add_argument(
            "--display",
            required=False,
            action="store_true",
            help="Choose whether you wish to display images.",
        )
        parser.add_argument(
            "--save-images",
            required=False,
            action="store_true",
            help="Choose whether you wish to save images.",
        )
        return parser


class MicroscopeParser(BaseParser):
    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--pixel-size",
            required=False,
            default=0.000024,
            type=float,
            help="Pixel width in centimeters.",
        )
        parser.add_argument(
            "--framestep-size",
            required=False,
            default=0.0217,
            type=float,
            help="Framestep size in seconds. Calculated using the FPS movie value.",
        )
        return parser


class ExportParser(BaseParser):
    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--save-location",
            required=False,
            default=None,
            type=str,
            help="Location where you wish to save the output table. Default location is wherever this script is ran from.",
        ),
        parser.add_argument(
            "--export-as",
            required=False,
            default="xlsx",
            type=str,
            help="Export file format extension. Defaults to 'xlsx'.",
        )
        return parser

    @staticmethod
    def set_save_location(parser_args: argparse.Namespace):
        if not parser_args.save_location:
            if parser_args.file:
                return os.path.split(parser_args.file)[0]
            return parser_args.directory


class VisualizationParser(BaseParser):
    @staticmethod
    def get_parser() -> argparse.ArgumentParser:
        parser = argparse.ArgumentParser(add_help=False)
        parser.add_argument(
            "--animate",
            required=False,
            action="store_true",
        )
        return parser
