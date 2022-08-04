import os
import io
import glob
import rasterio
import numpy as np
import pandas
from PIL import Image
from natsort import natsorted
from matplotlib import pyplot as plt
from matplotlib.patches import Patch
from scipy.optimize import curve_fit


def func_3_param(x, a, m, p):
    return ((a - p) * np.exp(-m * x)) + p


def func_2_param(x, a, m, p):
    return (a * np.exp(m * x)) - a


def fit_curve(x, y, fun):
    # Fitting
    popt, pcov = curve_fit(fun, x, y, p0=[1, 1, 1], maxfev=1000000)
    # R-squared
    residuals = y - fun(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return (*popt), r_squared


def make_gif(fps, fp_in, fp_out, stat_out):

    # Handle mobility dataframes
    full_dfs = [pandas.read_csv(fp) for fp in fps]
    full_dfs_clean = []
    for full_df in full_dfs:
        full_df_clean = pandas.DataFrame()
        for group, df in full_df.groupby('range'):
            df['x'] = df['year'] - df.iloc[0]['year']
            full_df_clean = full_df_clean.append(df)
        full_dfs_clean.append(full_df_clean)

    # Stack all blocks
    full_df = pandas.DataFrame()
    for df in full_dfs_clean:
        full_df = full_df.append(df)

    # Handle images
    imgs = [f for f in natsorted(glob.glob(fp_in))]
    years = {}
    for im in imgs:
        year = im.split('/')[-1].split('_')[1]
        print(year)
        years[str(year)] = im

    year_keys = list(full_df['year'].unique())
    years_filt = {}
    for key in year_keys:
        year_im = years.get(str(key), None)
        if not year_im:
            continue
        years_filt[key] = year_im

    years = years_filt
    year_keys = list(years.keys())
    imgs = []
    agrs = []
    combos = []
    for year, file in years.items():
        ds = rasterio.open(file).read(1)

        image = ds
        if year == year_keys[0]:
            agr = ds
        else:
            agr += ds

        ag_save = np.copy(agr)

        combo = agr + image

        imgs.append(image)
        agrs.append(ag_save)
        combos.append(combo)

    # Make avg_df
    avg_df = full_df.groupby('x').median().reset_index(drop=False).iloc[:20]
    avg_df = avg_df.dropna(how='any')

    # make max and min dfs
    max_df = full_df.groupby('x').max().reset_index(drop=False).iloc[:20]
    max_df = max_df.dropna(how='any')

    min_df = full_df.groupby('x').min().reset_index(drop=False).iloc[:20]
    min_df = min_df.dropna(how='any')

    am_3, m_3, pm_3, o_r2_3 = fit_curve(
        max_df['x'],
        max_df['O_Phi'].to_numpy(),
        func_3_param
    )

    ar_3, r_3, pr_3, f_r2_3 = fit_curve(
        min_df['x'],
        min_df['fR'].to_numpy(),
        func_3_param
    )

    stats = pandas.DataFrame(data={
        'Type': ['Value', 'Rsquared'],
        'M_3': [round(m_3, 8), round(o_r2_3, 8)],
        'AM_3': [round(am_3, 8), None],
        'PM_3': [round(pm_3, 8), None],
        'R_3': [round(r_3, 8), round(f_r2_3, 8)],
        'AR_3': [round(ar_3, 8), None],
        'PR_3': [round(pr_3, 8), None],
    })
    stats.to_csv(stat_out)

    o_pred_3 = func_3_param(avg_df['x'], am_3, m_3, pm_3)
    f_pred_3 = func_3_param(avg_df['x'], ar_3, r_3, pr_3)

    # METHOD 2
    images = []
    legend_elements = [
        Patch(color='#ad2437', label='Visited Pixels'),
        Patch(color='#6b2e10', label='Unvisted Pixels'),
        Patch(color='#9eb4f0', label='Yearly Water'),
    ]
    # for i, ag in enumerate(agrs):
    for i, ag in enumerate(combos):
        year = list(years.keys())[i]
        if i < len(avg_df):
            data = avg_df.iloc[i]
        img_buf = io.BytesIO()

        fig = plt.figure(constrained_layout=True, figsize=(10, 7))
        gs = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(gs[:, 0])
        ax2 = fig.add_subplot(gs[0, 1])
        ax3 = fig.add_subplot(gs[1, 1])

        ax1.imshow(ag, cmap='Paired_r')
        ax1.text(
            0.05,
            0.95,
            f'Year: {year}',
            horizontalalignment='left',
            verticalalignment='center',
            transform=ax1.transAxes
        )

        ax1.legend(
            handles=legend_elements,
            loc='lower left',
            prop={'size': 10}
        )

        ax2.plot(
            min_df['x'],
            f_pred_3,
            zorder=5,
            color='green',
            label='3 Parameter'
        )
        ax2.scatter(
            min_df['x'],
            min_df['fR'],
            zorder=4,
            s=70,
            facecolor='black',
            edgecolor='black'
        )
#        ax2.plot(
#            avg_df['x'],
#            avg_df['fR'],
#            zorder=4,
#            color='black',
#            label='average'
#        )
        ax2.scatter(
            full_df['x'],
            full_df['fR'],
            zorder=2,
            s=50,
            facecolor='white',
            edgecolor='black'
        )
        if i < len(avg_df):
            ax2.scatter(
                data['x'],
                data['fR'],
                s=200,
                zorder=3,
                color='red'
            )
        ax2.set_ylabel('Remaining Rework Fraction')
        ax2.legend(
            loc='upper left',
            frameon=True
        )

        ax3.plot(
            max_df['x'],
            o_pred_3,
            zorder=5,
            color='blue'
        )
        ax3.scatter(
            max_df['x'],
            max_df['O_Phi'],
            zorder=4,
            s=70,
            facecolor='black',
            edgecolor='black'
        )
#        ax3.plot(
#            avg_df['x'],
#            avg_df['O_Phi'],
#            zorder=4,
#            color='black',
#            label='average'
#        )
        ax3.scatter(
            full_df['x'],
            full_df['O_Phi'],
            zorder=2,
            s=50,
            facecolor='white',
            edgecolor='black'
        )
        ax3.scatter(data['x'], data['O_Phi'], s=200, zorder=3, color='red')
        ax3.set_ylabel('Normalized Channel Overlap')
        ax3.set_ylim([0, 1])

        plt.savefig(img_buf, format='png')
        images.append(Image.open(img_buf))
        plt.close('all')

    img, *imgs = images
    img.save(
        fp=fp_out,
        format='GIF',
        append_images=imgs,
        save_all=True,
        duration=400,
        loop=30
    )

    root = '/'.join(fps[0].split('/')[:-1])
    years = [i for i in range(1985, 2023)]
    for year in years:
        dire = os.path.join(root, str(year))
        if os.path.isdir(dire):
            os.rmdir(dire)


def make_gifs(river, root):
    print(river)
    fps = sorted(
        glob.glob(os.path.join(root, f'{river}/*mobility_block*.csv'))
    )
    fp_in = os.path.join(
        root, f'{river}/mask/*block_0.tif'
    )
    fp_out = os.path.join(
        root, f'{river}/{river}_cumulative.gif'
    )
    stat_out = os.path.join(
        root, f'{river}/{river}_mobility_stats.csv'
    )
    make_gif(fps, fp_in, fp_out, stat_out)
