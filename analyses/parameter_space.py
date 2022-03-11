import os
import glob
import pandas
from matplotlib import pyplot as plt


root = '/Users/Evan/Documents/Mobility/Parameter_space/Rivers'
fps = glob.glob(os.path.join(root, '*', '*stats.csv'))

meta_df = pandas.read_csv('/Users/Evan/Documents/Mobility/Parameter_space/RiverLocations.csv')

data = {
    'River': [],
    'M_3': [],
    'R_3': [],
    'M_2': [],
    'R_2': [],
    'Type': [],
}

for fp in fps:
    river = fp.split('/')[-2]
    if (river == 'Ganges1') or (river == 'Ganges2'):
        data['River'].append(river)
        river = 'Ganges'
    else:
        data['River'].append(river)
    print(river)
    river_type = meta_df[meta_df['River'] == river]
    if len(river_type) == 0:
        river_type = 'Delta'
    else:
        river_type = river_type['Type'].values[0]
    df = pandas.read_csv(fp)
    data['M_3'].append(df['M_3'].iloc[0])
    data['R_3'].append(df['R_3'].iloc[0])
    data['M_2'].append(df['M_2'].iloc[0])
    data['R_2'].append(df['R_2'].iloc[0])
    data['Type'].append(river_type)

param_df = pandas.DataFrame(data=data)

delta = param_df[param_df['Type'] == 'Delta']
braided = param_df[param_df['Type'] == 'Braided']
avulsion = param_df[param_df['Type'] == 'Avulsion']
meandering = param_df[param_df['Type'] == 'Meandering']


# plt.scatter(delta['M_3'], delta['R_3'], marker='v')
plt.scatter(braided['M_3'], braided['R_3'], marker='o', label='Braided', s=80, facecolor='red', edgecolor='black')
plt.scatter(avulsion['M_3'], avulsion['R_3'], marker='s', label='Avulsion', s=80, facecolor='blue', edgecolor='black')
plt.scatter(meandering['M_3'], meandering['R_3'], marker='*', label='Meandering', s=80, facecolor='green', edgecolor='black')
plt.legend()
plt.xlabel('M')
plt.ylabel('R')
plt.show()
