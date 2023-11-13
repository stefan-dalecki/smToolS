"""
Enums used throughout codebase.
"""

from enum import Enum, EnumMeta, StrEnum

__all__ = [
    "Cutoffs",
    "CutoffMethods",
    "Coordinates",
    "ReadFileTypes",
    "ExportFileTypes",
]


class EnumHelperMeta(EnumMeta):
    """
    Returns enum values using keys that might be incorrect case.
    """

    def __getitem__(cls, key: str):
        """
        Transform key to uppercase and strip periods before indexing.
        """

        return super().__getitem__(key.strip(".").upper())


class EnumHelper(Enum, metaclass=EnumHelperMeta):
    """
    Utility functions for getting enum class contents.
    """

    def __hash__(self):
        return hash(self.value)

    def __repr__(self):
        return str(self.value)

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
    """
    Discriminatory trajectory characteristics.
    """

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
    """
    Cartesian plane axes.
    """

    X = "x"
    Y = "y"


class FileTypeMeta(EnumHelper, StrEnum):
    """
    Quality of life parent class.
    """

    @staticmethod
    def sanitize(filetype: str):
        """
        Remove `.` and make string lowercase such that `.CSV` == `csv`
        """
        return filetype.strip(".").lower()

    def __contains__(self, filetype: str):
        return self.sanitize(filetype) in self.list_of_options()


class ReadFileTypes(FileTypeMeta):
    """
    Readable filetype extensions.
    """

    CSV = "csv"
    XML = "xml"
    ND2 = "nd2"


class ExportFileTypes(FileTypeMeta):
    CSV = "csv"
    XLSX = "xlsx"


class Proteins(EnumHelper, StrEnum):
    AKT = "akt"
    GRP = "grp"
    PDK = "pdk"
    PKC = "pkc"
    PX = "px"
