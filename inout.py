import pandas as pd
from tkinter import filedialog
from tkinter import Tk
import formulas as f


class Setup:
    """
    Initial program setup

    Establish a root directory and save file name

    """

    def __init__(self, file_format = 'csv'):
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

        root = Tk()
        root.withdraw()
        self.rootdir = filedialog.askdirectory()
        # self.rootdir = r'E:\PhD\Data\Johnny\ymd2022_02_10\gas1\Outputs'
        print(f'{self.rootdir}')
        self.savefile = input('Name your output summary file:\n')
        # self.savefile = 'test'
        self.options = ['csv', 'nd2']
        self.filetype = file_format

    def dfread(self, subdir, file):
        """
        Read in ND2 files

        loop through directory and find nd2 movie files

        Args:
            format (string): 3 character file format

        Returns:
            movie class object

        Raises:
            none

        """
        name = file[:-4]
        filepath = f'{subdir}/{file}'
        if self.filetype == 'csv':
            movie = Movie(name, filepath)
        if self.filetype == 'nd2':
            tracking = Track(filepath)
            movie = Movie(name, filepath, tracks = tracking.data)
        return movie


class Track:
    """
    Identify and link trajectories

    Use commercial, modified nearest neighbor algorithm to link particles

    """
    def __init__(self, im_path, quiet = False):
        """
        Track particles

        Use trackpy to identify and link individual particles

        Args:
            im_path (string): nd2 image file im_path
            quiet (bool): display text readout from trackpy

        Returns:
            trajectory data table as self.data

        Raises:
            none

        """

        from pims import ND2Reader_SDK
        import trackpy as tp

        with ND2Reader_SDK(im_path) as frames:
            low = min([i.min() for i in movie])
            diameter = 5
            pix_mov = 10
            if quiet:
                tp.quiet()
            f = tp.batch(movie, diameter, minmass = low,
                         output = None, processes = 20)
            t = tp.link(f, pix_mov)
        self.data = t

class Movie:
    """
    Movie object class

    Extract and establish data from image stack (movie) file

    """

    def __init__(self, name, file,
                 tracks = 'default', frame_cutoff=15):
        """
        Establish movie file parameters

        Define metadata related to movie and camera

        Args:
            name (string): name of movies
            file (string): rootdir location
            frame_cutoff (int): minimum frames for various analyses

        Returns:
            object metadata attributes

        Raises:
            none

        """

        self.name = {'Name': name}
        self.date = f.Find.identifiers(
            file, '/', 'Date Acquired', ['ymd'])
        self.protein_info = f.Find.identifiers(
            name, '_', 'Protein', ['grp', 'pdk'])
        self.gasket = f.Find.identifiers(file, '/', 'Gasket', ['gas'])
        self.replicate = {'Replicate': name[-2:]}
        self.ND = f.Find.identifiers(name, '_', 'ND Filter', ['nd'], '08')
        self.pixel_size = 0.000024
        self.framestep_size = 0.021
        self.frame_cutoff = frame_cutoff
        self.fig_title = f.Form.catdict(self.protein_info, self.ND,
                                        self.gasket, self.replicate)
        if tracks == 'default':
            self.df = pd.read_csv(file, index_col=[0])
        else:
            self.df = tracks
        self.exportdata = self.date | self.gasket | \
                          self.replicate | self.protein_info | self.ND
        print(f'Image Name : {name}',
              self.date, self.gasket, self.replicate, self.ND,
              f'Initial Trajectory Count : {f.Calc.traj_count(self.df)}',
              sep='\n', end='\n'*2)


class Exports:
    """
    Export data

    Save and ammend exported data

    """

    def __init__(self, name, exportdata, rootdir):
        """
        Export acquired data

        Build the dataframe and export all its data

        Args:
            name (string): typically the same as Setup.rootdir()
            export_df: takes keys (columns) from export data and creates empty
                dataframe

        Returns:
            none

        Raises:
            none

        """

        self.name = name
        self.export_df = pd.DataFrame(columns=list(exportdata.keys()))
        self.writer = rootdir+name+'.xlsx'

    def build_df(self, dict):
        """
        Generates the export dataframe

        Appends dictionary values to export_df

        Args:
            dict (dictionary): dictionary of movie kinetics/values

        Returns:
            export_df (df): dataframe containing data from all movies

        Raises:
            none

        """
        new_row = pd.DataFrame.from_dict(dict)
        self.export_df = pd.concat([self.export_df, new_row])

    def csv_export(self, df, rootdir):
        """
        Export dataframe as csv file

        can cause errors where the main script tries to read this file as image

        Args:
            df (df): final dataframe with all calculated data
            rootdir (string): directory location of save file

        Returns:
            csv file

        Raises:
            none

        """

        self.fidf = df
        df.to_csv(rootdir+f'\\{self.name}.csv', index=False)

    def xlsx_export(self, df, rootdir):
        """
        Export dataframe as xlsx file

        Tabs can be generated within this file to make it more digestable

        Args:
            df (df): final dataframe with all calculated data
            rootdir (string): directory location of save file

        Returns:
            xlsx file

        Raises:
            none

        """

        full_file = rootdir+f'\\{self.name}'
        self.fidf = df
        df.to_excel(full_file+'.xlsx', index=False, sheet_name='Raw')
        return full_file

    # def xlsx_write(self, kw):
    #     sheet_df = self.fdf.filter(regex=kw)
    #     with pd.ExcelWriter(self.full_file+'.xlsx') as writer:
    #         sheet_df.to_excel(writer, sheet_name=kw, index=False)
