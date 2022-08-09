"""Read in and out data"""

import os
import copy
import warnings
from collections import defaultdict
from tkinter import filedialog
import xml.etree.ElementTree as ET
import numpy as np
import pandas as pd
import formulas as fo
import cutoffs as cut


class MetaData:
    """Microscope parameters"""

    def __init__(
        self,
        *,
        pixel_size: float = 0.000024,
        framestep_size: float = 0.0217,
        frame_cutoff: int = 10,
    ) -> None:
        """Initialize microscope parameters

        Args:
            pixel_size (float, optional): pixel width in cm. Defaults to 0.000024.
            framestep_size (float, optional): time between frames. Defaults to 0.0217.
            frame_cutoff (int, optional): minimum trajectory length in frames. Defaults to 10.
        """
        # These values must be changed depending on your microscope qualities
        self.pixel_size = pixel_size
        self.framestep_size = framestep_size
        # frame_cutoff is dependent on biologic interest (i.e. are you interested
        # in proteins that are associated for a short or long period of time?)
        self.frame_cutoff = frame_cutoff

    def modify(self, **kwargs: dict[str, float or int]) -> object:
        """Temporarily modify metadata

        Returns:
            object: modified metadata
        """
        # Useful if you want to run different sub-routines within your program that
        # change the frame_cutoff or other characteristic
        temporary_metadata = copy.deepcopy(self)
        for kwarg in kwargs:
            for key, val in kwarg.items():
                setattr(temporary_metadata, key, val)
        return temporary_metadata


class Script:
    """
    Initial program setup

    Establish a root directory and save file name

    """

    # Excel is viewed as redundant and are read in slowly, it is not supported.
    # Any excel file can be converted to a csv with minimal difficulty.
    # Microsoft gets enough support already. Anyways...
    filetype_options = ["csv", "nd2", "hdf5", "xml"]

    def __init__(self) -> None:
        """
        Setup info for later export

        establishes the root directory and final output file name

        Args:
            none

        Returns:
            rootdir (string): directory for all major outputs
            savefile (string): name for final output (xlsx or csv, typically)

        Raises:
            none

        """
        self.filetype = None
        self.savefile = None
        self.rootdir = None
        self.cutoff_method = None
        self.booldisplay = False
        self.boolsave = False
        self.filelist = None

    def establish_constants(self) -> None:
        """Update script parameters"""
        self.filetype = fo.Form.userinput("filetype", Script.filetype_options)
        self.rootdir = filedialog.askdirectory()
        print(f"\nFile Directory chosen : {self.rootdir}")
        self.savefile = os.path.join(self.rootdir, input("\nName your output file: "))
        # Brightness options are based on brightness cutoff function names, with 'none'
        # as a way to bypass any brightness cutoffs. 'none' is necessary for xml files
        # from trackmate that do not contain any brightness data.
        brightness_options = (
            ["clustering"]
            + [
                option
                for option in dir(cut.Brightness)
                if option.startswith("__") is False
            ]
            + ["none"]
        )
        self.cutoff_method = fo.Form.userinput(
            "cutoff/brigthness thresholding method", brightness_options
        )
        if self.cutoff_method == "semi_auto":
            low_cut, high_cut = float(input("Low cutoff : ")), float(
                input("High cutoff : ")
            )
            self.cutoff_method = {"semi_auto": (low_cut, high_cut)}
        self.booldisplay = fo.Form.inputbool("Display figures?")
        self.boolsave = fo.Form.inputbool("Save figures?")

    def generate_filelist(self) -> None:
        """Build list of all files to analyze"""
        all_files = []
        for root, dirs, files in os.walk(self.rootdir):
            for name in files:
                if name.endswith(self.filetype):
                    all_files += [os.path.join(root, name)]
        all_files.sort()
        self.filelist = all_files

    @staticmethod
    def batching(task, filelist: list[str]) -> pd.DataFrame:
        """
        Batch process movies

        Analyze multiple movies simultaneously

        Args:
            task: processing function
            filelist (list): list of files to analyze

        Returns:
            concatenated dataframe

        Raises:
            Assert Error: if the number of processes is <= 1
            Assert Error: if filetype is not 'csv'

        """

        from multiprocessing import Pool

        dfs = []
        while True:
            print(f"Process -x- (max: {len(filelist)}) at once...")
            batch_size = int(input("x = ?: "))
            if batch_size > len(filelist):
                print("Error: processes > number of files")
            else:
                break
        while filelist:
            if batch_size >= len(filelist):
                batch_size = len(filelist)
            with Pool(batch_size) as pool:
                dfs += list(pool.map(task, filelist))
                pool.close()
                pool.join()
            filelist = filelist[batch_size:]
        return pd.concat(dfs, ignore_index=True)


class FileReader:
    """Reads raw datafile into workable table"""

    def __init__(self, filetype: str, filelist: list[str]) -> None:
        """Initialize file reader

        Args:
            filetype (str): filetype extension
            filelist (list): all file names to analyze
        """
        self.filetype = filetype
        self._rawfiles = filelist
        self.pre_processed_files = []
        reader_method = getattr(FileReader, f"process_{self.filetype}")
        reader_method(self)

    def process_csv(self) -> None:
        """Processes csv files"""
        for file in self._rawfiles:
            track_df = pd.read_csv(file, index_col=[0])
            # A quick check to see if the csv file contains trajectory data
            # or if it contains other nonsense
            if {"x", "y"} <= set(track_df.columns):
                # Some functions look for specifically named columns
                track_df.rename(
                    columns={
                        "particle": "Trajectory",
                        "frame": "Frame",
                        "m2": "Brightness",
                    },
                    inplace=True,
                )
                self.pre_processed_files += [(file, track_df)]

    def process_nd2(self) -> None:
        """Processes nd2 files"""
        from nd2reader import ND2Reader
        import trackpy as tp

        warnings.filterwarnings("ignore", category=UserWarning)
        tp.quiet()
        # This is a python specific method for particle tracking as opposed to
        # a method within ImageJ. For this reason, these three parameters must be
        # changed to satisfy your specific needs.
        DIAMETER = 9
        MOVEMENT = 10
        MEMORY = 1
        for file in self._rawfiles:
            h5_file_str = f"{file[:-4]}.h5"
            if not os.path.exists(h5_file_str):
                with ND2Reader(file) as movie:
                    low_mass = np.mean([np.median(i) for i in movie])
                    with tp.PandasHDFStoreBig(h5_file_str) as s:
                        print(
                            "\nBeginning ND2 Processing",
                            sep=2 * "\n",
                        )
                        for image in movie:
                            features = tp.locate(
                                image,
                                diameter=DIAMETER,
                                minmass=low_mass,
                                max_iterations=3,
                            )
                            s.put(features)
                        print(
                            "ND2 Processing Complete",
                            "Beginning Trajectory Linking",
                            sep=2 * "\n",
                        )
                        pred = tp.predict.NearestVelocityPredict()
                        for linked in pred.link_df_iter(
                            s,
                            search_range=MOVEMENT,
                            memory=MEMORY,
                            neighbor_strategy="BTree",
                        ):
                            s.put(linked)
                        track_df = pd.concat(s)
                        track_df = fo.Form.reorder(track_df, "x", 0)
                        track_df.rename(
                            columns={
                                "particle": "Trajectory",
                                "frame": "Frame",
                                "mass": "Brightness",
                            },
                            inplace=True,
                        )
                        track_df = track_df.sort_values(
                            ["Trajectory", "Frame"]
                        ).reset_index(drop=True)
                        print("Trajectory Linking Complete")
                        self.pre_processed_files += [(file, track_df)]

    def process_hdf5(self) -> None:
        """Read in an hdf5 file created by trackpy through an nd2"""
        for file in self._rawfiles:
            track_df = pd.concat(file)
            self.pre_processed_files += [(file, track_df)]

    def process_xml(self) -> None:
        """Read in an xml file generated by trackmate in ImageJ"""
        for file in self._rawfiles:
            root = ET.fromstring(open(file, encoding="utf-8").read())
            # Find column names based on first entry
            track_df = pd.DataFrame(columns=["Trajectory", *root[0][0].attrib.keys()])
            for i, val in enumerate(root):
                traj_dict = defaultdict(list)
                for j in val:
                    for k, v in j.attrib.items():
                        traj_dict[k].append(v)
                # Create and combine row based on dictionary of key value pairs
                traj_df = pd.DataFrame.from_dict(traj_dict).assign(Trajectory=i + 1)
                track_df = pd.concat([track_df, traj_df]).reset_index(drop=True)
            track_df.rename(columns={"t": "Frame"}, inplace=True)
            self.pre_processed_files += [file, track_df]


class Movie:
    """Individual movie object class

    Args:
        RawDataFrame (class): creates average trajectory brightness column
    """

    def __init__(
        self, metadata: object, filepath: str, trajectory_df: pd.DataFrame
    ) -> None:
        """Initialize movie object

        Args:
            metadata (object): persistent metadata
            filepath (string): full file location
            trajectory_df (pd.DataFrame): initial trajectory data
        """
        self.data_df = trajectory_df
        self.metadata = metadata
        self.filepath = os.path.normpath(filepath)
        self._name = {"FileName": self.filepath.split(os.sep)[-1][:-4]}
        self._date = fo.Find.date(filepath)
        self._gasket = fo.Find.identifiers(self.filepath, os.sep, "Gasket", ["gas"])
        self._replicate = {"Replicate": self._name["FileName"][-2:]}
        self._ND = fo.Find.identifiers(
            self._name["FileName"], "_", "ND Filter", ["nd"], failure="08"
        )
        self._protein = fo.Find.identifiers(
            self._name["FileName"], "_", "Protein", ["grp", "pdk", "pkc", "akt", "px"]
        )
        self.export_dict = {}
        self.export_dict["filepath"] = {filepath}
        # Only internal variables (self._) are used to build your output table
        for key, val in self.__dict__.items():
            if key.startswith("_"):
                for att_key, att_val in val.items():
                    self.export_dict[att_key] = att_val

        self.figure_title = copy.deepcopy(self._name["FileName"])

    def add_export_data(self, new_dict: dict) -> None:
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

    def save_df(self):
        """Saves trajectory data post-cutoffs"""

        self.data_df.to_excel(
            f"{self.filepath[:-4]}_preprocessed_data.xlsx",
            index=False,
            sheet_name="Keepers",
        )


class Export:
    """Export class"""

    export_file_types = ["csv", "xlsx"]
    # I recommend saving as an excel file for easier usage in excel
    # Also, I organize my work so that my csv's contain raw data
    # and my xlsx's contain results. But you do you.

    def __init__(self, script: object, df: pd.DataFrame) -> None:
        """Initialize export class object

        Args:
            script (class obect): script parameters
            df (pd.DataFrame): export dataframe
        """
        self._file_ext = fo.Form.userinput("filetype", Export.export_file_types)
        self._name = os.path.join(script.rootdir, f"{script.savefile}.{self._file_ext}")
        print(self._name)
        self._df = df

    def __call__(self) -> None:
        export_func = getattr(Export, self._file_ext)
        export_func(self)

    def csv(self) -> None:
        """Export data as csv"""
        try:
            self._df.to_csv(self._name, index=False)
            print("Export Successful")
        except Exception as e:
            print(f"Export Failed\nReason: {e}")

    def xlsx(self) -> None:
        """Export data as xlsx"""
        try:
            self._df.to_excel(self._name, index=False, sheet_name="Raw")
            print("Export Successful")
        except Exception as e:
            print(f"Export Failed\nReason: {e}")
