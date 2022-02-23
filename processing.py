import inout as io
import cutoffs as cut
import curvefitting as cf
import kinetics as k
import traceback
import os
from time import sleep


def main():
    """
    Main function for analyzing movies

    Central hub function that anaylyzes all movies in folder

    Args:
        bright_method (string): set method for brightness cutoff determination

    Returns:
        output file

    Raises:
        Exception: quit program

    """

    file_format = input('Analyze csv or nd2 files? (csv or nd2)\n')
    script = io.Setup(file_format = file_format)
    folder = script.rootdir
    outfile = script.savefile
    bright_method = input('Brightness thresholding method? (manual or auto)\n')
    display = False
    count = 1
    try:
        print('\nBeginning Analyses\n')
        for subdir, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith(script.filetype):
                    print(file)
                    file = file.replace('\\', '/')
                    subdir = subdir.replace('\\', '/')
                    print(subdir)
                    movie = script.dfread(subdir, file)
                    if bright_method == 'manual':
                        cut.Brightness.manual(movie, movie.df, display=display)
                    if bright_method == 'auto':
                        cut.Brightness.auto(movie, movie.df)
                    cut.Diffusion.displacement(movie, movie.df)
                    BSL = k.BSL(movie)
                    Ray = k.RayD(movie, movie.onestep_SDs)
                    MSD = k.MSD()
                    cf.FitFunctions.linear(movie, movie.mean_SDs,
                                           'Step Length', 'MSD', MSD)
                    cf.OneCompExpDecay(movie, BSL, display=display)
                    cf.TwoCompExpDecay(movie, BSL, display=display)
                    cf.Alt_TwoCompExpDecay(movie, BSL, display=display)
                    cf.ExpLinDecay(movie, BSL, display=display)
                    cf.TwoCompRayleigh(movie, Ray, display=display)
                    if count == 1:
                        todays_movies = io.Exports(
                            outfile, movie.exportdata, folder)
                        count = 0
                    todays_movies.build_df(movie.exportdata)
                    print(todays_movies.export_df)
    except Exception as e:
        print('Error', traceback.print_exc(e),
              'quitting program in 3 seconds.', sep='\n')
        sleep(3)
        quit()
    raw_excel = todays_movies.xlsx(todays_movies.export_df, folder)
    # todays_movies.xlsx_sheet()
    if raw_excel:
        print('Export successful', end='\n'*2)


if __name__ == '__main__':
    main()
    print('Analyses Complete\nEnding Program in...')
    count_down = 3
    for count in reversed(range(1, count_down+1)):
        print(count)
        sleep(1)
        pass
    quit()
