import simo_tools as simo

data = simo.DataFiles.from_path(
    r"sample_data\2023_02_05\gas1\Traj_67pM-GRP1_ND08_01.csv"
)
data.set_units(pixel_size=0.0024, fps=0.0217)
data.apply_cutoffs([
    simo.cutoffs.Brightness(min=1.2, max=4.2), simo.cutoffs.Length(min=5)
])
# result = data.analyze()
# result.save("my_path")

# figures = data.visualize()
# figures.save("my_path")
