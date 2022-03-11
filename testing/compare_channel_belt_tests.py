import copy
import os
import glob
import pandas 
from matplotlib import pyplot as plt
import numpy as np


water_root = '/Users/Evan/Documents/Mobility/GIS/Development/Channel_Belt_Box/water_based_channel_belt'
box_root = '/Users/Evan/Documents/Mobility/GIS/Development/Channel_Belt_Box/box_based_channel_belt'

water_fps = sorted(glob.glob(os.path.join(water_root, '*', '*stats.csv')))
box_fps = sorted(glob.glob(os.path.join(box_root, '*', '*stats.csv')))

data = {
    'River': [],
    'M_3': [],
    'R_3': [],
    'M_2': [],
    'R_2': [],
    'M_3_r2': [],
    'R_3_r2': [],
    'M_2_r2': [],
    'R_2_r2': [],
}
box_data = copy.deepcopy(data)
water_data = copy.deepcopy(data)
for water_fp, box_fp in zip(water_fps, box_fps):
    river = water_fp.split('/')[-2]
    print(river)
    water_df = pandas.read_csv(water_fp)
    box_df = pandas.read_csv(box_fp)

    water_data['River'].append(river)
    water_data['M_3'].append(water_df.iloc[0]['M_3'])
    water_data['R_3'].append(water_df.iloc[0]['R_3'])
    water_data['M_2'].append(water_df.iloc[0]['M_2'])
    water_data['R_2'].append(water_df.iloc[0]['R_2'])
    water_data['M_3_r2'].append(water_df.iloc[1]['M_3'])
    water_data['R_3_r2'].append(water_df.iloc[1]['R_3'])
    water_data['M_2_r2'].append(water_df.iloc[1]['M_2'])
    water_data['R_2_r2'].append(water_df.iloc[1]['R_2'])

    box_data['River'].append(river)
    box_data['M_3'].append(box_df.iloc[0]['M_3'])
    box_data['R_3'].append(box_df.iloc[0]['R_3'])
    box_data['M_2'].append(box_df.iloc[0]['M_2'])
    box_data['R_2'].append(box_df.iloc[0]['R_2'])
    box_data['M_3_r2'].append(box_df.iloc[1]['M_3'])
    box_data['R_3_r2'].append(box_df.iloc[1]['R_3'])
    box_data['M_2_r2'].append(box_df.iloc[1]['M_2'])
    box_data['R_2_r2'].append(box_df.iloc[1]['R_2'])

water_stats = pandas.DataFrame(data=water_data)
box_stats = pandas.DataFrame(data=box_data)

water_root = '/Users/Evan/Documents/Mobility/GIS/Development/Channel_Belt_Box/water_based_channel_belt'
box_root = '/Users/Evan/Documents/Mobility/GIS/Development/Channel_Belt_Box/box_based_channel_belt'

water_fps = sorted(glob.glob(os.path.join(water_root, '*', '*mobility.csv')))
box_fps = sorted(glob.glob(os.path.join(box_root, '*', '*mobility.csv')))

fig, axs = plt.subplots(3, 2, sharex=True, sharey=True, tight_layout=True)
axs = np.ravel(axs)
for i, (water_fp, box_fp) in enumerate(zip(water_fps, box_fps)):
    river = water_fp.split('/')[-2]
    water_df = pandas.read_csv(water_fp)
    box_df = pandas.read_csv(box_fp)

    water_df_clean = pandas.DataFrame()
    for group, df in water_df.groupby('range'):
        df = df.reset_index(drop=True)
        df['x'] = df['year'] - df.iloc[0]['year']
        water_df_clean = water_df_clean.append(df)
    water_df = water_df_clean
    water_df_clean = None

    box_df_clean = pandas.DataFrame()
    for group, df in box_df.groupby('range'):
        df = df.reset_index(drop=True)
        df['x'] = df['year'] - df.iloc[0]['year']
        box_df_clean = box_df_clean.append(df)
    box_df = box_df_clean
    box_df_clean = None

    water_df = water_df.groupby('x').mean().reset_index(drop=False)
    box_df = box_df.groupby('x').mean().reset_index(drop=False)

    axs[i].plot(water_df['x'], water_df['O_Phi'], color='green', zorder=1, label='Water')
    axs[i].scatter(water_df['x'], water_df['O_Phi'], s=20, facecolor='white', edgecolor='green', zorder=2)
    axt = axs[i].twinx()
    axt.plot(box_df['x'], box_df['O_Phi'], color='blue', zorder=1, label='Box')
    axt.scatter(box_df['x'], box_df['O_Phi'], s=20, facecolor='white', edgecolor='blue', zorder=2)

    axs[i].set_title(river)
    axs[i].set_ylabel('O_Phi')

    water_r3 = water_stats[water_stats['River'] == river].iloc[0]['R_3']
    water_r2 = water_stats[water_stats['River'] == river].iloc[0]['R_2']
    water_m3 = water_stats[water_stats['River'] == river].iloc[0]['M_3']
    water_m2 = water_stats[water_stats['River'] == river].iloc[0]['M_2']

    box_r3 = box_stats[box_stats['River'] == river].iloc[0]['R_3']
    box_r2 = box_stats[box_stats['River'] == river].iloc[0]['R_2']
    box_m3 = box_stats[box_stats['River'] == river].iloc[0]['M_3']
    box_m2 = box_stats[box_stats['River'] == river].iloc[0]['M_2']

    water_r3_r2 = water_stats[water_stats['River'] == river].iloc[0]['R_3_r2']
    water_r2_r2 = water_stats[water_stats['River'] == river].iloc[0]['R_2_r2']
    water_m3_r2 = water_stats[water_stats['River'] == river].iloc[0]['M_3_r2']
    water_m2_r2 = water_stats[water_stats['River'] == river].iloc[0]['M_2_r2']

    box_r3_r2 = box_stats[box_stats['River'] == river].iloc[0]['R_3_r2']
    box_r2_r2 = box_stats[box_stats['River'] == river].iloc[0]['R_2_r2']
    box_m3_r2 = box_stats[box_stats['River'] == river].iloc[0]['M_3_r2']
    box_m2_r2 = box_stats[box_stats['River'] == river].iloc[0]['M_2_r2']

    axs[i].text(
        .45,
        .05,
        f'Water 3-Param M - R2: {water_m3} - {water_m3_r2}',
        horizontalalignment='right', 
        verticalalignment='center', 
        transform=axs[i].transAxes
    )
    axs[i].text(
        .45,
        .1,
        f'Box 3-Param M - R2: {box_m3} - {box_m3_r2}',
        horizontalalignment='right', 
        verticalalignment='center', 
        transform=axs[i].transAxes
    )
    axs[i].text(
        .45,
        .15,
        f'Water 2-Param M - R2: {water_m2} - {water_m2_r2}',
        horizontalalignment='right', 
        verticalalignment='center', 
        transform=axs[i].transAxes
    )
    axs[i].text(
        .45,
        .2,
        f'Box 2-Param M - R2: {box_m2} - {box_m2_r2}',
        horizontalalignment='right', 
        verticalalignment='center', 
        transform=axs[i].transAxes
    )


axs[-2].legend()
axt.legend()
plt.show()





