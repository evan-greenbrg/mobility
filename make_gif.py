import io
import os
import glob
import rasterio
import numpy as np
import pandas
from PIL import Image
from PIL import ImageFont
from PIL import ImageDraw 
from natsort import natsorted
from matplotlib import pyplot as plt
from matplotlib import cm
import matplotlib.gridspec as gridspec
from matplotlib.patches import Patch

# filepaths
fp = sorted(glob.glob('/Users/Evan/Documents/Mobility/GIS/Taiwan/Taiwan3/*.csv'))[0]
fp_in = "/Users/Evan/Documents/Mobility/GIS/Taiwan/Taiwan3/temps/*.tif"
fp_out = "/Users/Evan/Documents/Mobility/GIS/Taiwan/Taiwan3/taiwan3_cumulative.gif"

# https://pillow.readthedocs.io/en/stable/handbook/image-file-formats.html#gif
imgs = [f for f in natsorted(glob.glob(fp_in))]
df = pandas.read_csv(fp)

years = {}
for im in imgs:
    year = im.split('/')[-1].split('_')[1].split('.')[0]
    if years.get(year):
        years[year].append(im)
    else:
        years[year] = im

year_keys = list(df['year'])
years_filt = {}
for key in year_keys:
    years_filt[key] = years[str(key)]
years = years_filt
year_keys = list(years.keys())
imgs = []
agrs = []
combos = []
for year, file in years.items():
    ds = rasterio.open(file).read(1)
    ds[ds > 1] = 999
    ds[ds < 900] = 0
    ds[ds == 999] = 1

    image = ds
    if year == year_keys[0]:
        agr = ds
    else:
        agr += ds

    agr[agr > 0] = 999
    agr[agr < 900] = 0
    agr[agr == 999] = 1

    ag_save = np.copy(agr)

    combo = agr + image

    imgs.append(image)
    agrs.append(ag_save)
    combos.append(combo)

# METHOD 2
images = []
legend_elements = [
    Patch(color='#ad2437', label='Visited Pixels'),
    Patch(color='#6b2e10', label='Unvisted Pixels'),
    Patch(color='#9eb4f0', label='Yearly Water'),
]
#for i, ag in enumerate(agrs):
for i, ag in enumerate(combos):
    data = df.iloc[i]
    img_buf = io.BytesIO()

    fig = plt.figure(constrained_layout=True, figsize=(10,7))
    gs = fig.add_gridspec(3, 2)
    ax1 = fig.add_subplot(gs[:, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 1])
    ax4 = fig.add_subplot(gs[2, 1])

#    ax1.imshow(ag, cmap='Greys_r')
    ax1.imshow(ag, cmap='Paired_r')
    ax1.legend(handles=legend_elements, loc='lower left', prop={'size': 10})

    ax2.plot(df['year'], 1- df['fR'], zorder=1, color='black')
    ax2.scatter(df['year'], 1- df['fR'], zorder=2, color='black')
    ax2.scatter(data['year'], 1- data['fR'], s=200, zorder=3, color='red')
    ax2.set_ylim([0,1.1])
    ax2.set_ylabel('Remaining Rework Fraction')

    ax3.scatter(df['year'], df['O_Phi'], zorder=2, color='black')
    ax3.scatter(data['year'], data['O_Phi'], s=200, zorder=3, color='red')
    ax3.set_ylabel('Normalized Channel Overlap')

    ax4.scatter(df['year'], df['zeta'], zorder=2, color='black')
    ax4.scatter(data['year'], data['zeta'], s=200, zorder=3, color='red')
    ax4.set_ylabel('Mobility Rate')
    ax4.set_xlabel('Year')
#    plt.show()

    plt.savefig(img_buf, format='png')
    images.append(Image.open(img_buf))

plt.close('all')

img, *imgs = images 
img.save(fp=fp_out, format='GIF', append_images=imgs,         save_all=True, duration=400, loop=1)
