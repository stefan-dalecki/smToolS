import display as d
import numpy as np
import pandas as pd
import formulas as f
from collections import defaultdict
from statistics import mean


class Brightness:
    def manual(movie, df, display=False):
        print('Beginning Pre-Processing...')
        AVG = df.groupby('Trajectory')['m2'].mean()
        df = df.join(AVG, on='Trajectory', rsuffix='-Average')
        while True:
            d.Histogram.basic(AVG.values, 70,
                              movie.name,
                              'Intensity (m2)',
                              'Number of Trajectories')
            low_out = float(input('Select the low brightness cutoff : '))
            high_out = float(input('Select the high brightness cutoff : '))
            not_low = df[df['m2-Average'] > low_out]
            not_high = df[df['m2-Average'] < high_out]
            list_low = not_low['Trajectory'].values
            list_high = not_high['Trajectory'].values
            rm_outliers_df = df[df['Trajectory'].isin(list_low)]
            rm_outliers_df = rm_outliers_df[rm_outliers_df['Trajectory'].isin(
                list_high)]
            rm_df = rm_outliers_df.reset_index(drop=True)
            grp_df = rm_df.groupby('Trajectory') \
                .filter(lambda x: len(x) > movie.frame_cutoff) \
                .reset_index(drop=True)
            print(f.Calc.traj_count(grp_df))
            move_on = input('Choose new cutoffs (0) or continue? (1) : ')
            if move_on == '1':
                break
        # for trajectory in df['Trajectory'].unique():
        #     t_rows = df[df['Trajectory'] == trajectory]
        #     df = df.drop(min(t_rows.index))
        movie.alldf = rm_df
        movie.df = grp_df
        movie.exportdata.update({'Cutoffs': (low_out, high_out)})
        print('PreProcessing Complete')

    def auto(movie, df, display=False):
        AVG = df.groupby('Trajectory')['m2'].mean()
        df = df.join(AVG, on='Trajectory', rsuffix='-Average')
        step = 1/10
        bin_sdf = np.arange(0, 7, step)
        groups = 2
        while groups <= 10:
            single_traj = df.drop_duplicates(subset='Trajectory', keep='first')
            sdf = single_traj. \
                groupby(pd.cut(single_traj['m2-Average'], bins=bin_sdf)) \
                .size().nlargest(groups)
            cutoff_list = [i.right and i.right for i in sdf.index]
            low_out, high_out = round(
                min(cutoff_list), 1), round(max(cutoff_list), 1)
            not_low = df[df['m2-Average'] > low_out]
            not_high = df[df['m2-Average'] < high_out]
            list_low = not_low['Trajectory'].values
            list_high = not_high['Trajectory'].values
            rm_outliers_df = df[df['Trajectory'].isin(list_low)]
            rm_outliers_df = rm_outliers_df[rm_outliers_df['Trajectory']
                                            .isin(list_high)]
            rm_df = rm_outliers_df.reset_index(drop=True)
            grp_df = rm_df.groupby('Trajectory') \
                .filter(lambda x: len(x) > movie.frame_cutoff) \
                .reset_index(drop=True)
            if f.Calc.traj_count(grp_df) > 150:
                break
            else:
                groups += 1
                continue
        if display:
            d.Histogram.lines(AVG.values, 70, low_out, high_out,
                              movie.name,
                              'Intensity (m2)',
                              'Number of Trajectories')
        # for trajectory in df['Trajectory'].unique():
        #     t_rows = df[df['Trajectory'] == trajectory]
        #     df = df.drop(min(t_rows.index))
        movie.alldf = rm_df
        movie.df = grp_df
        movie.exportdata.update({'Cutoffs': (low_out, high_out)})


class Diffusion:
    def displacement(movie, df):
        print('Beginning Diffusion Calculations...')
        movie.all_SDs = defaultdict(list)
        movie.mean_SDs = pd.DataFrame(columns=['Step Length', 'MSD'])
        movie.onestep_SDs = []
        for trajectory in df['Trajectory'].unique():
            SD_SL1 = []
            t_rows = df[df['Trajectory'] == trajectory]
            x_col = t_rows['x'].values
            y_col = t_rows['y'].values
            if len(t_rows - 3) > 8:
                max_step_len = 8
            else:
                max_step_len = len(t_rows - 3)
            for step_len in range(1, max_step_len + 1):
                for step_num in range(0, len(t_rows) - step_len - 1):
                    x1, y1 = x_col[step_num], y_col[step_num]
                    x2, y2 = x_col[step_num
                                   + step_len], y_col[step_num + step_len]
                    squared_distance = f.Calc.distance(x1, y1, x2, y2) ** 2
                    if step_len == 1:
                        SD_SL1.append(squared_distance)
                if step_len == 1:
                    diff_coeff1 = mean(
                        SD_SL1) * movie.pixel_size**2 / \
                        (4*movie.framestep_size)
                    if diff_coeff1 <= 5e-9 or diff_coeff1 >= 5e-8:
                        movie.df.drop(
                            df.loc[df['Trajectory'] == trajectory].index,
                            inplace=True)
                        break
                    else:
                        movie.onestep_SDs += [i**(1/2)*movie.pixel_size
                                              for i in SD_SL1]
                        movie.all_SDs[step_len].append(squared_distance)
                else:
                    movie.all_SDs[step_len].append(squared_distance)
        for i in range(1, max_step_len+1):
            mean_steplen = mean(
                movie.all_SDs[i]) * movie.pixel_size**2 / \
                (4*movie.framestep_size)
            new_row = pd.DataFrame.from_dict({'Step Length': [i],
                                              'MSD': [mean_steplen]})
            movie.mean_SDs = pd.concat([movie.mean_SDs, new_row])
        print(
            f'\nFinal Trajectory Count : {f.Calc.traj_count(movie.df)}')
        movie.exportdata.update({'Events': len(movie.df)})
        movie.exportdata.update(
            {'Traj': f.Calc.traj_count(movie.df)})
