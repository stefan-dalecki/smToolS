"""
String constants used throughout codebase.
"""

__all__ = [
    "AVERAGE_BRIGHTNESS",
    "BASE_TRAJECTORY_TABLE_COLS",
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
    "SEC",
    "SDS",
    "TRAJECTORY",
    "TRAJECTORY_W_UNITS",
]


AVERAGE_BRIGHTNESS = "average_brightness"
BRIGHTNESS = "brightness"
DATE = "date"
FILENAME = "filename"
FILEPATH = "filepath"
FRAME = "frame"
GASKET = "gasket"
GASKET_ABV = "gas"
LENGTH_W_UNITS = "length (frames)"
M2 = "m2"
MASS = "mass"
MSD = "msd"
MSD_W_UNITS = f"{MSD} (\u03bcm\u00b2/sec)"
ND_FILTER = "nd_filter"
ND_FILTER_ABV = "nd"
PARTICLE = "particle"
PRE_PROCESSED = "pre_processed"
PROTEIN = "protein"
REPLICATE = "replicate"
SDS = "sds"
SEC = "sec"
TRAJECTORY = "trajectory"
TRAJECTORY_W_UNITS = "trajectories (#)"
X = "x"
Y = "y"

BASE_TRAJECTORY_TABLE_COLS = [
    X,
    Y,
    TRAJECTORY,
    FRAME,
    BRIGHTNESS,
]
