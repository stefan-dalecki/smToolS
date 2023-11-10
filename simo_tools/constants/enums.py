"""
Enums used throughout codebase.
"""

from enum import Enum, StrEnum

__all__ = [
    "Cutoffs",
    "CutoffMethods",
    "Coordinates",
    "ReadFileTypes",
    "ExportFileTypes",
]


class EnumHelper(Enum):
    """
    Utility functions for getting enum class contents.
    """

    @classmethod
    def list_of_options(cls) -> list[str]:
        """
        All enum values as list.
        """
        return [c.value for c in cls]

    @classmethod
    def set_of_options(cls) -> set[str]:
        """
        All enum values as set.
        """
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


class FileTypeMeta(EnumHelper, StrEnum):
    @staticmethod
    def sanitize(filetype: str):
        return filetype.strip(".").lower()

    def __contains__(self, filetype: str):
        return self.sanitize(filetype) in self.list_of_options()


class ReadFileTypes(FileTypeMeta):
    CSV = "csv"
    H5 = "h5"
    ND2 = "nd2"
    XML = "xml"


class ExportFileTypes(FileTypeMeta):
    CSV = "csv"
    XLSX = "xlsx"


class Proteins(EnumHelper, StrEnum):
    AKT = "akt"
    GRP = "grp"
    PDK = "pdk"
    PKC = "pkc"
    PX = "px"
