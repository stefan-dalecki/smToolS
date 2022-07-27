import os
import pandas as pd

file_dir = r"D:\PhD\Data\Stefan\Best_GRP1\Pre_Processed"

file = r"four_grp1_raw.xlsx"

df = pd.read_excel(os.path.join(file_dir, file))


def keepers(df):
    keep_df = df[df["Average_Brightness"].between(3.1, 3.8)]
    keep_df = keep_df[keep_df["Length (frames)"] > 10]
    keep_df = keep_df[keep_df["MSD"].between(0.3, 2.5)]
    return keep_df


good = keepers(df)
print(good.shape[0])
