"""Read in and out data"""
import copy
import dataclasses
import logging
import os
import re
import warnings
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime
from typing import AnyStr, Dict, List, LiteralString, Optional, Self, Tuple

import numpy as np
import pandas as pd
import trackpy as tp
from nd2reader import ND2Reader

from smToolS.analysis_tools import formulas as fo
from smToolS.sm_helpers import constants as cons

logging.basicConfig()
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)


def find_date(
    full_str: str,
    *,
    failure: str = "no date found",
    date_format: tuple[str, str] = (r"\d{4}_\d{2}_\d{2}", "%Y_%m_%d"),
) -> Dict[str, datetime.date]:
    """Find the date from a string (i.e. file path)

    Args:
        full_str (str): full string to search
        failure (str, optional): what do return when no date is found.
            Defaults to "no date found".
        date_format (tuple[str, str], optional): the date format for searching.
            Defaults to (r"\d{4}_\d{2}_\d{2}", "%Y_%m_%d").

    Returns:
        dict: _description_
    """
    # \d{4}_\d{2}_\d{2} is the same as saying 1234_12_12, four numbers followed by
    # two numbers, followed by another two numbers. Month and day are not distinguished.
    # %Y-%m-%d is the same as Year-Month-Day. This takes the number above as separates
    # The first four numbers into the year, second two into the month, and last two
    # into the day.
    output = {}
    match_str = re.search(date_format[0], full_str)
    if match_str:
        output["date"] = datetime.strptime(match_str.group(), date_format[1]).date()
    else:
        output["date"] = failure
    return output


def find_identifiers(
    full_string: AnyStr,
    separator: AnyStr,
    value_name: AnyStr,
    value_search_names: List[AnyStr],
    *,
    failure: LiteralString = "not found",
) -> Dict[AnyStr, AnyStr]:
    """Find image attributes from file name

    Args:
        full_string (AnyStr): full string in question
        separator (AnyStr): separator between sub string elements
        value_name (AnyStr): value of interest
        value_search_names (List[AnyStr]): potential value names
        failure (LiteralStr, optional): what to return if nothing is found. Defaults to "not found".

    Raises:
        RuntimeError: nothing is found

    Returns:
        Dict: search result as dictionary
    """
    # Files named as follows can easily be interpreted by this function
    # Data\Stefan\2021\2021_11_02\gas1\67pM-GRP1_ND06_01
    output = {}
    separated_full_string = [i for i in full_string.lower().split(separator)]
    value_search_names = [name.lower() for name in value_search_names]
    for search_name in value_search_names:
        hit_indeces = [
            i for i, val in enumerate(separated_full_string) if val.find(search_name) != -1
        ]
        for hit_index in hit_indeces:
            region = separated_full_string[hit_index]
            if os.path.sep in region:
                break
            if "-" in region:
                if "+" in region:
                    protein_descriptions = region.split("+")
                else:
                    protein_descriptions = [region]
                for protein_description in protein_descriptions:
                    concentration, protein = protein_description.split("-")
                    protein = protein.upper()
                    concentration_value = concentration[:-2]
                    concentration_units = concentration[-2:]
                    output[f"{protein} ({concentration_units})"] = concentration_value
            else:
                output[value_name] = region.replace(str(search_name), "")
                break
    if not output:
        output[value_name] = failure
    return output


def reorder(df: pd.DataFrame, column_name: AnyStr, location: int) -> pd.DataFrame:
    """Reorder dataframe columns

    Args:
        df (pd.DataFrame): dataframe
        column_name (AnyStr): name of column to move
        location (int): new column location index

    Returns:
        pd.DataFrame: rearrange dataframe
    """
    if column_name not in df.columns:
        raise KeyError(f"Column '{column_name}' not found in dataframe columns: '{df.columns}'.")
    column_data = df[column_name].values
    df = df.drop(columns=[column_name])
    df.insert(location, column_name, column_data)
    return df


def _attribute_validation(item: AnyStr, options: List[AnyStr]):
    if item not in options:
        message = f"'{item}' is not valid. Options include: {options}"
        raise KeyError(message)


# I wish this was frozen but when manually setting cutoffs, we do need to change this class here
@dataclasses.dataclass
class Script:
    """
    Setup info for usage throughout script

    """

    filetype: Optional[str] = None
    file: Optional[str] = None
    directory: Optional[str] = None
    display: Optional[bool] = None
    save_images: Optional[bool] = None
    cutoffs: Optional[List[str]] = None
    brightness_method: Optional[str] = None
    min_length: Optional[int] = None
    # diffusion_method: str
    brightness_cutoffs: Optional[Tuple[float, float]] = None
    diffusion_cutoffs: Optional[Tuple[float, float]] = None

    def __post_init__(self):
        self._validate_cutoffs()
        self._validate_brightness_method()
        if self.filetype:
            self._validate_filetype()
        else:
            self.filetype = os.path.splitext(self.file)[1][1:]  # return file extension without '.'

    @property
    def cutoff_options(self):
        return cons.Cutoffs.list_of_options()

    @property
    def brightness_method_options(self):
        return cons.CutoffMethods.list_of_options()

    @property
    def filetype_options(self) -> List[str]:
        return cons.FileTypes.read_filetypes()

    @property
    def file_list(self):
        all_files = []
        if not self.directory and self.file:
            return [self.file]  # just return file in list if we are not looking at a directory
        for root, dirs, files in os.walk(self.directory):
            for name in files:
                if name.endswith(self.filetype):
                    all_files += [os.path.join(root, name)]
        all_files.sort()
        return all_files

    def _validate_cutoffs(self):
        if not self.cutoffs:
            return
        for cutoff in self.cutoffs:
            _attribute_validation(cutoff, self.cutoff_options)

    def _validate_brightness_method(self):
        if not self.brightness_method:
            return
            # xml files cannot specify a brightness method
        if self.filetype == cons.FileTypes.XML and self.brightness_method != "none":
            message = "Invalid brightness cutoff method for 'xml' filetype. Method must be 'None'."
            raise ValueError(message)
        # the semi_auto method must be paired with already specified brightness cutoffs
        if self.brightness_method == "semi_auto" and not self.brightness_cutoffs:
            message = "Brightness cutoffs must be specified when using 'semi_auto' cutoff method."
            raise ValueError(message)
        _attribute_validation(self.brightness_method, self.brightness_method_options)

    def _validate_filetype(self):
        _attribute_validation(self.filetype, self.filetype_options)

    def _validate_cutoff_values(self):
        for val in (*self.brightness_cutoffs, *self.diffusion_cutoffs):
            if val < 0.0:
                raise ValueError(f"'{val}' : Cutoff value cannot be negative.")


class Microscope:
    """Microscope parameters / movie parameters"""

    def __init__(
        self,
        pixel_size: float,
        framestep_size: float,
    ) -> None:
        """Initialize microscope parameters

        These values are dependent on the qualities of your microscope
        Args:
            pixel_size (float, optional): pixel width in cm. Defaults to 0.000024.
            framestep_size (float, optional): time between frames. Defaults to 0.0217.
        """
        self.pixel_size = pixel_size
        self.framestep_size = framestep_size

    def modify(self, **kwargs: Dict[str, any]) -> Self:
        """Temporarily modify metadata

        Useful if you want to run different sub-routines within your program that
        change the frame_cutoff or other characteristic
        Returns:
            object: modified metadata
        """
        temporary_metadata = copy.deepcopy(self)
        for key, val in kwargs.items():
            setattr(temporary_metadata, key, val)
        return temporary_metadata

    def calc_all_steps_no_min(self, df: pd.DataFrame) -> defaultdict[list]:
        df = df.reset_index(drop=True)
        x_col, y_col = df[cons.Coordinates.X], df[cons.Coordinates.Y]
        all_steps = defaultdict(list)
        max_step_len = 8
        for step_len in range(1, max_step_len + 1):
            for step_num in range(0, len(df) - step_len - 1):
                x1, y1 = x_col[step_num], y_col[step_num]
                x2, y2 = x_col[step_num + step_len], y_col[step_num + step_len]
                distance = (
                    fo.calc_distance(x1, y1, x2, y2) ** 2
                    * self.pixel_size**2
                    / (4 * self.framestep_size)
                )
                all_steps[step_len].append(distance)
        return all_steps

    def calc_one_step_MSD(self, df: pd.DataFrame) -> Tuple[Optional[List], Optional[List]]:
        """Calculate preliminary mean squared displacement

        Args:
            df (pd.DataFrame): data for one trajectory

        Returns:
            tuple: mean squared displacement
        """
        df = df.reset_index(drop=True)
        x_col, y_col = df[cons.Coordinates.X], df[cons.Coordinates.Y]
        if len(df) > 1:
            SDs = [np.nan]
            for i in range(len(df) - 1):
                x1, y1 = x_col[i], y_col[i]
                x2, y2 = x_col[i + 1], y_col[i + 1]
                squared_distance = fo.calc_distance(x1, y1, x2, y2) ** 2
                SDs.append(squared_distance)
            MSD = np.nanmean(SDs) * self.pixel_size**2 / (4 * self.framestep_size) * 1e8
            return SDs, MSD
        else:
            return None, None


class FileReader:
    """Reads raw datafile into workable table"""

    def __init__(
        self, filetype: Optional[str], filelist: Script.file_list, framestep: float
    ) -> None:
        """Initialize file reader

        Args:
            filetype (str): filetype extension
            filelist (List[str]): all file names to analyze
        """
        self.filetype = filetype
        self._rawfiles = filelist
        self._framestep = framestep
        self.pre_processed_files = []  # populate using process_<filetype> method
        self.reader_method = getattr(self, f"_process_{self.filetype}")
        self()

    def __call__(self, *args, **kwargs):
        logger.info(f"Beginning '{self.filetype}' file processing.")
        self.reader_method()
        logger.info(f"Completed '{self.filetype}' file processing.")

    @property
    def one_table(self) -> pd.DataFrame:
        logger.info(f"Combining '{[file[0] for file in self.pre_processed_files]}' into one table.")
        tables = [i[1] for i in self.pre_processed_files]
        return pd.concat(tables)

    @staticmethod
    def _sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
        """Make column names are lower case characters
        Args:
            df (pd.DataFrame): dataframe

        Returns:
            pd.DataFrame: all column names are completely lowercase
        """
        df.columns = df.columns.str.lower()
        return df

    def _process_csv(self) -> None:
        """Processes csv files"""
        for file in self._rawfiles:
            track_df = pd.read_csv(file, index_col=[0])
            # A quick check to see if the csv file contains trajectory data
            # or if it contains other nonsense
            if cons.PRE_PROCESSED not in file and cons.Coordinates.set_of_options() <= set(
                track_df.columns
            ):
                # Some functions look for specifically named columns
                track_df.rename(
                    columns={
                        cons.PARTICLE: cons.TRAJECTORY,
                        cons.M2: cons.Cutoffs.BRIGHTNESS,
                    },
                    inplace=True,
                )
                track_df = self._sanitize_column_names(track_df)
                self.pre_processed_files += [(file, track_df)]

    def _process_nd2(self) -> None:
        """Processes nd2 files"""
        warnings.filterwarnings("ignore", category=UserWarning)
        tp.quiet()
        # This is a python specific method for particle tracking as opposed to
        # a method within ImageJ. For this reason, these three parameters must be
        # changed to satisfy your specific needs.
        DIAMETER = 9
        MOVEMENT = 10
        MEMORY = 1
        ND2 = cons.FileTypes.ND2
        for file in self._rawfiles:
            h5_file_str = f"{file[:-4]}.h5"
            if not os.path.exists(h5_file_str):
                with ND2Reader(file) as movie:
                    low_mass = np.mean([np.median(i) for i in movie]) * 1.5
                    with tp.PandasHDFStoreBig(h5_file_str) as s:
                        logger.info(f"Beginning '{ND2}' Processing")
                        for image in movie:
                            features = tp.locate(
                                image,
                                diameter=DIAMETER,
                                minmass=low_mass,
                            )
                            s.put(features)
                        logger.info(f"'{ND2}' Processing Complete")
                        logger.info("Beginning Trajectory Linking")
                        pred = tp.predict.NearestVelocityPredict()
                        for linked in pred.link_df_iter(
                            s,
                            search_range=MOVEMENT,
                            memory=MEMORY,
                            neighbor_strategy="BTree",
                        ):
                            s.put(linked)
                        # for linked in tp.link_df_iter(
                        #     s, search_range=MOVEMENT, memory=MEMORY
                        # ):
                        # s.put(linked)
                        track_df = pd.concat(s)
                        track_df = reorder(track_df, "x", 0)
                        track_df.rename(
                            columns={
                                cons.PARTICLE: cons.TRAJECTORY,
                                # cons.Frame: "c",
                                cons.MASS: cons.BRIGHTNESS,
                            },
                            inplace=True,
                        )
                        track_df = track_df[
                            track_df.groupby(cons.TRAJECTORY)[cons.TRAJECTORY].transform("count")
                            > 1
                        ]
                        track_df = track_df.sort_values([cons.TRAJECTORY, cons.FRAME]).reset_index(
                            drop=True
                        )
                        track_df = reorder(track_df, cons.TRAJECTORY, 0)
                        track_df = reorder(track_df, cons.FRAME, 1)
                        track_df = self._sanitize_column_names(track_df)
                        self.pre_processed_files += [(file, track_df)]
                        logger.info("Trajectory Linking Complete")

    def _process_h5(self) -> None:
        """Read in an hdf5 file created by trackpy through an nd2"""
        for file in self._rawfiles:
            with tp.PandasHDFStoreBig(file) as s:
                track_df = pd.concat(s)
                track_df = reorder(track_df, "x", 0)
                track_df.rename(
                    columns={
                        cons.PARTICLE: cons.TRAJECTORY,
                        # "frame": "Frame",
                        cons.MASS: cons.BRIGHTNESS,
                    },
                    inplace=True,
                )
                track_df = track_df.sort_values([cons.TRAJECTORY, cons.FRAME]).reset_index(
                    drop=True
                )
                track_df = reorder(track_df, cons.TRAJECTORY, 0)
                track_df = reorder(track_df, cons.FRAME, 1)
                track_df = track_df[
                    track_df.groupby(cons.TRAJECTORY)[cons.TRAJECTORY].transform("count") > 1
                ]
                track_df = self._sanitize_column_names(track_df)
                self.pre_processed_files += [(file, track_df)]

    def _process_xml(self) -> None:
        """Read in an xml file generated by trackmate in ImageJ"""
        for file in self._rawfiles:
            root = ET.fromstring(open(file, encoding="utf-8").read())
            # Find column names based on first entry
            track_df = pd.DataFrame(columns=[cons.TRAJECTORY, *root[0][0].attrib.keys()])
            for i, val in enumerate(root):
                traj_dict = defaultdict(list)
                for j in val:
                    for k, v in j.attrib.items():
                        traj_dict[k].append(v)
                # Create and combine row based on dictionary of key value pairs
                traj_df = pd.DataFrame.from_dict(traj_dict).assign(Trajectory=i + 1)
                track_df = pd.concat([track_df, traj_df]).reset_index(drop=True)
            track_df.rename(columns={"t": cons.FRAME}, inplace=True)
            track_df[cons.Coordinates.X], track_df[cons.Coordinates.Y] = track_df[
                cons.Coordinates.X
            ].astype(float), track_df[cons.Coordinates.Y].astype(float)
            track_df[cons.Coordinates.X] *= self._framestep**-1
            track_df[cons.Coordinates.Y] *= self._framestep**-1
            track_df = self._sanitize_column_names(track_df)
            self.pre_processed_files += [(file, track_df)]


class Movie:
    """Individual movie object class"""

    def __init__(self, script_metadata: Script, filepath: str, trajectory_df: pd.DataFrame) -> None:
        """Initialize movie object

        Args:
            script_metadata (Script): persistent metadata
            filepath (str): full file location
            trajectory_df (pd.DataFrame): initial trajectory data
        """
        self.metadata = script_metadata
        self.filepath = os.path.normpath(filepath)
        self.data_df = trajectory_df
        self.name = {cons.FILENAME: os.path.splitext(os.path.basename(self.filepath))[0]}
        self._date = find_date(filepath)
        self._gasket = find_identifiers(filepath, os.sep, cons.GASKET, [cons.GASKET_ABV])
        self._replicate = {cons.REPLICATE: self.name[cons.FILENAME][-2:]}
        self._ND = find_identifiers(
            self.name[cons.FILENAME], "_", cons.ND_FILTER, [cons.ND_FILTER_ABV], failure="08"
        )
        self._protein = find_identifiers(
            self.name[cons.FILENAME], "_", cons.PROTEIN, cons.Proteins.list_of_options()
        )
        self.export_dict = self._init_export_dict(filepath)
        self.figure_title = copy.deepcopy(self.name[cons.FILENAME])

    def _init_export_dict(self, filepath: str) -> Dict:
        """
        Create the initial export dictionary
        """
        # Only private variables (self._) are used to further build your output table
        export_dict = {cons.FILEPATH: filepath}
        for key, val in self.__dict__.items():
            if key.startswith("_"):
                for att_key, att_val in val.items():
                    export_dict[att_key] = att_val
        return export_dict

    def add_export_data(self, new_dict: Dict) -> None:
        """Builds export dictionary

        Args:
            new_dict (dict): key (column) and values (cell) for export
        """
        self.export_dict.update(new_dict)

    def update_trajectory_df(self, *, new_df: pd.DataFrame) -> None:
        """Updates data_df attribute

        Args:
            new_df (pd.DataFrame): updated (thresholded) trajectory data
        """
        setattr(self, "data_df", new_df)

    def save_df(self, prefix: AnyStr):
        """Saves trajectory data post-cutoffs"""
        NAME = os.path.splitext(self.filepath)[0]
        self.data_df.to_excel(
            f"{NAME}_{prefix}_data.xlsx",
            index=False,
            sheet_name="Keepers",
        )


class Export:
    """Export class"""

    export_file_types = cons.FileTypes.export_filetypes

    # I recommend saving as an excel file for easier usage in excel
    # Also, I organize my work so that my csv's contain raw data
    # and my xlsx's contain results. But you do you.

    def __init__(
        self,
        df: pd.DataFrame,
        save_location: AnyStr,
        export_filetype: Optional[cons.FileTypes.export_filetypes],
        save_filename: Optional[AnyStr] = "summary",
    ) -> None:
        """Initialize export class object

        Args:
            save_filename (AnyStr): output savefile name
            df (pd.DataFrame): export dataframe
            export_filetype (cons.FileTypes.export_filetypes): file extension
            save_location (AnyStr): directory where file will be saved
        """
        self._file_ext = export_filetype
        self._name = os.path.join(save_location, f"{save_filename}.{self._file_ext}")
        self._df = df
        self()

    def __call__(self) -> None:
        logger.info(f"Attempting export of '{self._name}'")
        export_func = getattr(Export, self._file_ext)
        export_func(self)
        logger.info(f"Export successful. Filetype: '{self._file_ext}'")

    def csv(self) -> None:
        """Export data as csv"""
        self._df.to_csv(self._name, index=False)

    def xlsx(self) -> None:
        """Export data as xlsx"""
        self._df.to_excel(self._name, index=False, sheet_name="Raw")
