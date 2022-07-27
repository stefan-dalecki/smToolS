"""Read in and out data"""

import os
import copy
import warnings
from tkinter import filedialog
from tkinter import *
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
        frame_cutoff: int = 9,
    ) -> None:
        """Initialize microscope parameters

        Args:
            pixel_size (float, optional): pixel width. Defaults to 0.000024.
            framestep_size (float, optional): time between frames. Defaults to 0.0217.
            frame_cutoff (int, optional): minimum trajectory length in frames. Defaults to 10.
        """
        self.pixel_size = pixel_size
        self.framestep_size = framestep_size
        self.frame_cutoff = frame_cutoff

    def modify(self, **kwargs: dict[str, float or int]) -> object:
        """Temporarily modify metadata

        Returns:
            object: modified metadata
        """
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
        self._filetype_options = ["csv", "nd2", "hdf5"]
        self.filetype = None
        self.savefile = None
        self.rootdir = None
        self.parallel_process = False
        self.cutoff_method = None
        self.booldisplay = False
        self.boolsave = False
        self.filelist = None

    def establish_constants(self) -> None:
        """Update script parameters"""
        self.filetype = fo.Form.userinput("filetype", self._filetype_options)
        root = Tk()
        root.withdraw()
        self.rootdir = filedialog.askdirectory()
        print(f"\nFile Directory chosen : {self.rootdir}")
        self.savefile = os.path.join(self.rootdir, input("\nName your output file: "))
        if self.parallel_process:
            self.cutoff_method = "auto"
            self.booldisplay = False
        else:
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
            # self.boolprint = fo.Form.inputbool("Print progress?")
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
        self._filetype = filetype
        self._rawfiles = filelist
        self.pre_processed_files = []
        reader_method = getattr(FileReader, f"process_{self._filetype}")
        reader_method(self)

    def process_csv(self) -> None:
        """Processes csv files"""
        for file in self._rawfiles:
            track_df = pd.read_csv(file, index_col=[0])
            if {"x", "y"} <= set(track_df.columns):
                track_df.rename(
                    columns={
                        "paricle": "Trajectory",
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

    def process_hdf5(self):
        for file in self._rawfiles:
            track_df = pd.concat(file)
            self.pre_processed_files += [(file, track_df)]


class Movie(RawDataFrame):
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
        self._date = fo.Find.identifiers(
            self.filepath, os.sep, "Date Acquired", ["ymd"]
        )
        self._gasket = fo.Find.identifiers(self.filepath, os.sep, "Gasket", ["gas"])
        self._replicate = {"Replicate": self._name["FileName"][-2:]}
        self._ND = fo.Find.identifiers(
            self._name["FileName"], "_", "ND Filter", ["nd"], failure="08"
        )
        self._protein = fo.Find.identifiers(
            self._name["FileName"], "_", "Protein", ["grp", "pdk", "pkc", "akt", "px"]
        )
        self.export_dict = {}
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
        """Saves trajectory dataframe for machine learning cases"""

        self.data_df.to_excel(
            f"{self.filepath[:-4]}_preprocessed_data.xlsx",
            index=False,
            sheet_name="Keepers",
        )


class Export:
    """Export class"""

    export_file_types = ["csv", "xlsx"]

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
