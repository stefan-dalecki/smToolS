import inout as io
import cutoffs as cut
import curvefitting as cf
import kinetics as k
import traceback
import os
from time import sleep


def main(script):
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

    count = 1
    try:
        for file in script.filelist:
            movie = script.dfread(file)
            if script.brightmethod == 'manual':
                cut.Brightness.manual(movie, movie.df, script.display)
            if script.brightmethod == 'auto':
                cut.Brightness.auto(movie, movie.df, script.display)
            cut.Diffusion.displacement(movie, movie.df)
            BSL = k.BSL(movie)
            Ray = k.RayD(movie, movie.onestep_SDs)
            MSD = k.MSD()
            cf.FitFunctions.linear(movie, movie.mean_SDs,
                                   'Step Length', 'MSD', MSD)
            cf.OneCompExpDecay(movie, BSL, script.display)
            cf.TwoCompExpDecay(movie, BSL, script.display)
            cf.Alt_TwoCompExpDecay(movie, BSL, script.display)
            cf.ExpLinDecay(movie, BSL, script.display)
            cf.OneCompRayleigh(movie, Ray, script.display)
            cf.TwoCompRayleigh(movie, Ray, script.display)
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
    script = io.Setup()
    script.filelist()
    folder = script.rootdir
    outfile = script.savefile
    main(script)
    print('Analyses Complete\nEnding Program in...')
    count_down = 3
    for count in reversed(range(1, count_down+1)):
        print(count)
        sleep(1)
        pass
    quit()
