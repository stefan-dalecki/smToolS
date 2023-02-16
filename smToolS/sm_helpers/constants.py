from enum import StrEnum

__all__ = [
    "AVERAGE_BRIGHTNESS",
    "BRIGHTNESS",
    "DATE",
    "FILENAME",
    "FILEPATH",
    "FRAME",
    "GASKET",
    "GASKET_ABV",
    "LENGTH_W_UNITS",
    "MASS",
    "MSD",
    "MSD_W_UNITS",
    "M2",
    "ND_FILTER",
    "ND_FILTER_ABV",
    "PARTICLE",
    "PRE_PROCESSED",
    "PROTEIN",
    "REPLICATE",
    "SDS",
    "TRAJECTORY",
    "TRAJECTORY_W_UNITS",
    "CutoffMethods",
    "FileTypes",
    "Proteins",
]

from typing import List, Set

AVERAGE_BRIGHTNESS = "average_brightness"
BRIGHTNESS = "brightness"
DATE = "date"
FILENAME = "filename"
FILEPATH = "filepath"
FRAME = "frame"
GASKET = "gasket"
GASKET_ABV = "gas"
LENGTH_W_UNITS = "length (frames)"
MASS = "mass"
MSD = "msd"
MSD_W_UNITS = "msd (\u03bcm\u00b2/sec)"
M2 = "m2"
ND_FILTER = "nd_filter"
ND_FILTER_ABV = "nd"
PARTICLE = "particle"
PRE_PROCESSED = "pre_processed"
PROTEIN = "protein"
REPLICATE = "replicate"
SDS = "sds"
TRAJECTORY = "trajectory"
TRAJECTORY_W_UNITS = "trajectories (#)"


class EnumHelper:
    @classmethod
    def list_of_options(cls) -> List[str]:
        return [c.value for c in cls]

    @classmethod
    def set_of_options(cls) -> Set[str]:
        return {c.value for c in cls}


class Cutoffs(EnumHelper, StrEnum):
    BRIGHTNESS = "brightness"
    LENGTH = "length"
    DIFFUSION = "diffusion"


class CutoffMethods(EnumHelper, StrEnum):
    MANUAL = "manual"
    SEMI_AUTO = "semi_auto"
    AUTO = "auto"
    CLUSTERING = "clustering"

    @classmethod
    def default(cls):
        return None


class Coordinates(EnumHelper, StrEnum):
    X = "x"
    Y = "y"


class FileTypes(EnumHelper, StrEnum):
    CSV = "csv"
    H5 = "h5"
    ND2 = "nd2"
    XLSX = "xlsx"
    XML = "xml"

    @classmethod
    def read_filetypes(cls) -> List[str]:
        return [cls.CSV, cls.H5, cls.ND2, cls.XML]

    @classmethod
    def export_filetypes(cls) -> List[str]:
        return [cls.CSV, cls.XLSX]


class Proteins(EnumHelper, StrEnum):
    AKT = "akt"
    GRP = "grp"
    PDK = "pdk"
    PKC = "pkc"
    PX = "px"
