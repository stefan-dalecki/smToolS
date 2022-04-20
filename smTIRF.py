import cutoffs as cut
import curvefitting as cf
import formulas as f
import inout as io
import kinetics as k
import traceback
from time import sleep
import itertools
import pandas as pd


def main(file, script):
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

    try:
        movie = script.dfread(file)
        if script.brightmethod == "manual":
            cut.Brightness.manual(movie, movie.df, script.display)
        if script.brightmethod == "auto":
            cut.Brightness.auto(movie, movie.df, script.display)
        cut.Diffusion.displacement(movie, movie.df)
        BSL = k.BSL(movie)
        Ray = k.RayD(movie, movie.onestep_SDs)
        MSD = k.MSD()
        cf.FitFunctions.linear(movie, movie.mean_SDs, "Step Length", "MSD", MSD)
        cf.OneCompExpDecay(movie, BSL, script.display)
        cf.TwoCompExpDecay(movie, BSL, script.display)
        cf.Alt_TwoCompExpDecay(movie, BSL, script.display)
        cf.ExpLinDecay(movie, BSL, script.display)
        cf.OneCompRayleigh(movie, Ray, script.display)
        cf.TwoCompRayleigh(movie, Ray, script.display)
        cf.ThreeCompRayleigh(movie, Ray, script.display)

    except Exception as e:
        print(
            "Error", traceback.print_exc(e), "quitting program in 3 seconds.", sep="\n"
        )
        sleep(3)
        quit()

    return pd.DataFrame(movie.exportdata, index=[0])


if __name__ == "__main__":
    script = io.Setup()
    script.filelist()
    folder = script.rootdir
    outfile = script.savefile
    export = io.Exports(script.savefile, script.rootdir)
    if script.parallel:
        while True:
            print(f"Process -x- (max: {len(script.filelist)}) at once...")
            par = int(input("x = ?: "))
            if par > len(script.filelist):
                print("Error: processes > number of files")
            else:
                break
        dfs = f.Form.batching(main, script, export, par, script.filelist)
    else:
        dfs = pd.concat(map(main, script.filelist, itertools.repeat(script)))
    excel_doc = export.xlsx(script, dfs)
    count_down = 3
    for count in reversed(range(1, count_down + 1)):
        print(count)
        sleep(1)
        pass
    quit()
