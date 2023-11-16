import simo_tools as simo

simo.Meta(pixel_size=0.00024, framestep_size=0.0217)

data = simo.DataFiles.from_path(
    r"sample_data\2023_02_05\gas1\Traj_67pM-GRP1_ND08_01.csv"
)
