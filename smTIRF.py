"""Central module"""

import traceback
import sys
from time import sleep
import pandas as pd
import cutoffs as cut
import curvefitting as cf
import inout as io
import kinetics as kin
import formulas as fo


def main(file: tuple) -> pd.DataFrame:
    """Main processing pipeline

    Args:
        file (tuple): filename and filedata tuple

    Returns:
        pd.DataFrame: export data for specific movie
    """

    try:
        if script.boolprint:
            print("\nBEGINNING Analysis\n")

        movie_path, trajectories = file
        movie = io.Movie(metadata, movie_path, trajectories)
        movie.update_trajectory_df(new_df=fo.Calc.trio(metadata, movie.data_df))

        if script.boolprint:
            print(f"   Beginning ---{script.cutoff_method}--- Method Cutoffs\n")

        if script.cutoff_method == "none":
            minimum_length = cut.Length(metadata, movie.data_df, method="minimum")
            movie.update_trajectory_df(new_df=minimum_length.cutoff_df)

        elif script.cutoff_method == "clustering":
            cluster = cut.Clustering(metadata, movie.data_df)
            cluster.scale_features().estimate_clusters().display().cluster()
            metadata.__setattr__("frame_cutoff", cluster.min_length)
            movie.update_trajectory_df(new_df=cluster.cutoff_df)

        else:
            brightness = cut.Brightness(metadata, movie.data_df, script.cutoff_method)
            movie.update_trajectory_df(new_df=brightness.cutoff_df)
            minimum_length = cut.Length(metadata, movie.data_df, method="minimum")
            movie.update_trajectory_df(new_df=minimum_length.cutoff_df)

        diffusion = cut.Diffusion(metadata, movie.data_df)
        movie.update_trajectory_df(new_df=diffusion.cutoff_df)
        movie.add_export_data(
            {"Trajectories (#)": fo.Calc.trajectory_count(movie.data_df)}
        )

        if script.boolprint:
            print("   Constructing kinetics\n")

        bsl = (
            kin.Director(kin.BSL(metadata, movie.data_df))
            .construct_kinetic()
            .get_kinetic()
        )
        msd = (
            kin.Director(kin.MSD(metadata, movie.data_df))
            .construct_kinetic()
            .get_kinetic()
        )
        movie.add_export_data(kin.MSD.model(msd.table))
        rayd = (
            kin.Director(kin.RayD(metadata, movie.data_df["SDs"].dropna()))
            .construct_kinetic()
            .get_kinetic()
        )

        if script.boolprint:
            print("   Kinetics constructed\n")

        if script.boolprint:
            print("   Building models")

        OneCompExpDecay = (
            cf.Director(
                cf.ExpDecay(metadata, movie, components=1, kinetic=bsl, table=bsl.table)
            )
            .build_model()
            .get_model()
        )
        movie.add_export_data(OneCompExpDecay.dictify())

        TwoCompExpDecay = (
            cf.Director(
                cf.ExpDecay(metadata, movie, components=2, kinetic=bsl, table=bsl.table)
            )
            .build_model()
            .get_model()
        )
        movie.add_export_data(TwoCompExpDecay.dictify())

        OneCompRayleigh = (
            cf.Director(
                cf.RayDiff(
                    metadata, movie, components=1, kinetic=rayd, table=rayd.table
                )
            )
            .build_model()
            .get_model()
        )

        movie.add_export_data(OneCompRayleigh.dictify())

        TwoCompRayleigh = (
            cf.Director(
                cf.RayDiff(
                    metadata, movie, components=2, kinetic=rayd, table=rayd.table
                )
            )
            .build_model()
            .get_model()
        )
        movie.add_export_data(TwoCompRayleigh.dictify())

        if script.boolprint:
            print(
                "   Models built\n",
                "Analysis COMPLETE\n",
                sep="\n",
            )

        if script.booldisplay or script.boolsave:
            OneCompExpDecay.generate_plot(script.booldisplay, script.boolsave)
            TwoCompExpDecay.generate_plot(script.booldisplay, script.boolsave)
            OneCompRayleigh.generate_plot(script.booldisplay, script.boolsave)
            TwoCompRayleigh.generate_plot(script.booldisplay, script.boolsave)

    except RuntimeError as error:
        print(
            "Error",
            traceback.print_exc(error),
            "quitting program in 3 seconds.",
            sep="\n",
        )
        sleep(3)
        sys.exit()

    return pd.DataFrame(movie.export_dict, index=[0])


if __name__ == "__main__":
    script = io.Script()
    metadata = io.MetaData()
    script.establish_constants()
    script.generate_filelist()
    file_dictionary = io.FileReader(script.filetype, script.filelist)
    if script.parallel_process:
        dfs = io.Script.batching(main, file_dictionary.pre_processed_files)
    else:
        dfs = pd.concat(map(main, file_dictionary.pre_processed_files))
    export = io.Export(script, dfs)
    export()
    count_down = 3
    for count in reversed(range(1, count_down + 1)):
        print(count)
        sleep(1)
    sys.exit()
