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
from scipy.optimize import curve_fit


def func(x, a, m, p):
    return ((a - p) * np.exp(-m * x)) + p


def fit_curve(x, y):
    # Fitting
    popt, pcov = curve_fit(func, x, y, p0=[1, .00001, 0], maxfev=1000)
    # R-squared
    residuals = y - func(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return (*popt), r_squared

def main(fp, fp_in, fp_out, stat_out):
    imgs = [f for f in natsorted(glob.glob(fp_in))]
    df = pandas.read_csv(fp)

    years = {}
    for im in imgs:
        year = im.split('_')[-1].split('.')[0]
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

    # Get Curve fits 
    df['x'] = df['year'] - df.iloc[0]['year']
    am, m, pm, o_r2 = fit_curve(
        df['x'],
        df['O_Phi'].to_numpy()
    )

    ar, r, pr, f_r2 = fit_curve(
        df['x'],
        1 - df['fR'].to_numpy()
    )
    zeta_median = df['zeta'].median()

    stats = pandas.DataFrame(data={
        'Type': ['Value', 'Rsquared'],
        'M': [round(m, 4), round(o_r2, 4)],
        'R': [round(r, 4), round(f_r2, 4)],
        'zeta': [round(zeta_median, 4), None],
    })
    stats.to_csv(stat_out)

    o_pred = func(df['x'], am, m, pm)
    f_pred = func(df['x'], ar, r, pr)

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

        ax2.plot(df['year'], f_pred, zorder=1, color='blue')
        ax2.scatter(df['year'], 1- df['fR'], zorder=2, color='black')
        ax2.scatter(data['year'], 1- data['fR'], s=200, zorder=3, color='red')
        ax2.set_ylim([0,1.1])
        ax2.set_ylabel('Remaining Rework Fraction')
        ax2.text(2012, .95, f'R: {round(r, 2)}')
        ax2.text(2012, .85, f'r-squared: {round(f_r2, 2)}')

        ax3.plot(df['year'], o_pred, zorder=1, color='blue')
        ax3.scatter(df['year'], df['O_Phi'], zorder=2, color='black')
        ax3.scatter(data['year'], data['O_Phi'], s=200, zorder=3, color='red')
        ax3.set_ylabel('Normalized Channel Overlap')
        ax3.text(2012, .95, f'M: {round(m, 2)}')
        ax3.text(2012, .85, f'r-squared: {round(o_r2, 2)}')

        ax4.plot(df['year'], [zeta_median for i in df['year']])
        ax4.scatter(df['year'], df['zeta'], zorder=2, color='black')
        ax4.scatter(data['year'], data['zeta'], s=200, zorder=3, color='red')
        ax4.set_ylabel('Mobility Rate')
        ax4.set_xlabel('Year')
        ax4.text(1990, 0, f'Zeta: {round(zeta_median, 2)}')
    #    plt.show()

        plt.savefig(img_buf, format='png')
        images.append(Image.open(img_buf))

    plt.close('all')

    img, *imgs = images 
    img.save(fp=fp_out, format='GIF', append_images=imgs,         save_all=True, duration=400, loop=1)

# Rivers
# X Apalachicola_River
# X Beatton
# X Chickasawhay
# X Escambia
# Indus
# Kazak1
# Milk_River
# Minnesota_River_Downstream
# Minnesota_River_Upststream
# Ocmulgee
# Okavango_River
# Pearl
# Pembina
# Ramganga_River
# Sacramento_River
# Strickland_River
# Tarim
# Wabash_River
# Watut_River
# White_River
# Yellow_River

rivers = [
    'Kazak1',
    'Milk_River',
    'Minnesota_River_Downstream',
    'Minnesota_River_Upstream',
    'Ocmulgee',
    'Okavango_River',
    'Pearl',
    'Pembina',
    'Ramganga_River',
    'Sacramento_River',
    'Strickland_River',
    'Tarim',
    'Wabash_River',
    'Watut_River',
    'White_River',
    'Yellow_River',
]
for river in rivers:
    print()
    print(river)
    print()
    fp = sorted(glob.glob(f'//home/greenberg/ExtraSpace/PhD/Projects/Mobility/GIS/Comparing/Rivers/{river}/*mobility.csv'))[0]
    fp_in = f'/home/greenberg/ExtraSpace/PhD/Projects/Mobility/GIS/Comparing/Rivers/{river}/temps/*.tif'
    fp_out = f'/home/greenberg/ExtraSpace/PhD/Projects/Mobility/GIS/Comparing/Rivers/{river}/{river}_cumulative.gif'
    stat_out = f'/home/greenberg/ExtraSpace/PhD/Projects/Mobility/GIS/Comparing/Rivers/{river}/{river}_mobility_stats.csv'

    main(fp, fp_in, fp_out, stat_out)

