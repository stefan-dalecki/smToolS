import inout as io
import cutoffs as cut
import curvefitting as cf
import kinetics as k
import traceback
import os
from time import sleep

def main(bright_method = 'auto'):
    count = 1
    script = io.Setup()
    folder = script.rootdir
    outfile = script.savefile
    try:
        print('\nBeginning Analyses\n')
        for subdir, dirs, files in os.walk(folder):
            for file in files:
                if file.endswith('.csv'):
                    print(file)
                    file = file.replace('\\', '/')
                    subdir = subdir.replace('\\', '/')
                    print(subdir)
                    movie = io.Movie(file[:-4], f'{subdir}/{file}')
                    if bright_method == 'manual':
                        cut.Brightness.manual(movie, movie.df, display = False)
                    if bright_method == 'auto':
                        cut.Brightness.auto(movie, movie.df)
                    cut.Diffusion.displacement(movie, movie.df)
                    BSL = k.BSL(movie)
                    Ray = k.RayD(movie, movie.onestep_SDs)
                    MSD = k.MSD()
                    cf.FitFunctions.linear(movie, movie.mean_SDs,
                                           'Step Length', 'MSD', MSD)
                    cf.OneCompExpDecay(movie, BSL, display = False)
                    cf.TwoCompExpDecay(movie, BSL, display = False)
                    cf.Alt_TwoCompExpDecay(movie, BSL, display = False)
                    cf.ExpLinDecay(movie, BSL, display = False)
                    cf.TwoCompRayleigh(movie, Ray, display = False)
                    if count == 1:
                        todays_movies = io.Exports(
                            outfile, movie.exportdata, folder)
                        count = 0
                    todays_movies.build_df(movie.exportdata)
                    print(todays_movies.export_df)
    except Exception as e:
        print('Error', traceback.print_exc(e), 'quitting program in 3 seconds.', sep='\n')
        sleep(3)
        quit()
    raw_excel = todays_movies.xlsx_export(todays_movies.export_df, folder)
    todays_movies.xlsx_sheet()
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
