import glob
import argparse
import os
import ee
import ee.mapclient
import pandas
import numpy as np
from matplotlib import pyplot as plt
from natsort import natsorted
import io
import rasterio
import numpy as np
from PIL import Image
from matplotlib.patches import Patch
from scipy.optimize import curve_fit

from puller_fun import pull_esa
from puller_fun import clean_esa
from puller_fun import create_mask_shape
from puller_fun import clean_channel_belt 
from puller_fun import filter_images
# from puller_fun import get_mobility_yearly


# ee.Authenticate()
ee.Initialize()


def make_gif(fp, fp_in, fp_out, stat_out):
    imgs = [f for f in natsorted(glob.glob(fp_in))]
    full_df = pandas.read_csv(fp)

    years = {}
    for im in imgs:
        year = im.split('_')[-1].split('.')[0]
        if years.get(year):
            years[year].append(im)
        else:
            years[year] = im

    year_keys = list(full_df['year'].unique())
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
    full_df_clean = pandas.DataFrame()
    for group, df in full_df.groupby('range'):
        df['x'] = df['year'].astype(int) - int(df.iloc[0]['year'])
        full_df_clean = full_df_clean.append(df)

    full_df = full_df_clean
    full_df_clean = None
#    full_df = full_df.dropna(subset=['x', 'O_Phi', 'fR'])

    avg_df = full_df.groupby('x').mean().iloc[:20].reset_index(drop=False)

    am_3, m_3, pm_3, o_r2_3 = fit_curve(
        avg_df['x'],
        avg_df['O_Phi'].to_numpy(),
        func_3_param
    )

    ar_3, r_3, pr_3, f_r2_3 = fit_curve(
        avg_df['x'],
        avg_df['fR'].to_numpy(),
#        avg_df['fR'].to_numpy(),
        func_2_param_pos
    )

    am_2, m_2, pm_2, o_r2_2 = fit_curve(
        avg_df['x'],
        avg_df['O_Phi'].to_numpy(),
        func_2_param
    )

    ar_2, r_2, pr_2, f_r2_2 = fit_curve(
        avg_df['x'],
        avg_df['fR'].to_numpy(),
#        avg_df['fR'].to_numpy(),
        func_2_param_pos
    )

    stats = pandas.DataFrame(data={
        'Type': ['Value', 'Rsquared'],
        'M_3': [round(m_3, 8), round(o_r2_3, 8)],
        'AM_3': [round(am_3, 8), None],
        'PM_3': [round(pm_3, 8), None],
        'R_3': [round(r_3, 8), round(f_r2_3, 8)],
        'AR_3': [round(ar_3, 8), None],
        'PR_3': [round(pr_3, 8), None],
        'M_2': [round(m_2, 8), round(o_r2_2, 8)],
        'AM_2': [round(ar_2, 8), None],
        'PM_2': [round(pr_2, 8), None],
        'R_2': [round(r_2, 8), round(f_r2_2, 8)],
        'AR_2': [round(ar_2, 8), None],
        'PR_2': [round(pr_2, 8), None],
    })

    return stats, avg_df


def test_equations(fp, fp_in, fp_out, stat_out):
    imgs = [f for f in natsorted(glob.glob(fp_in))]
    full_df = pandas.read_csv(fp)

    years = {}
    for im in imgs:
        year = im.split('_')[-1].split('.')[0]
        if years.get(year):
            years[year].append(im)
        else:
            years[year] = im

    year_keys = list(full_df['year'].unique())
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
    full_df_clean = pandas.DataFrame()
    for group, df in full_df.groupby('range'):
        df['x'] = df['year'].astype(int) - int(df.iloc[0]['year'])
        full_df_clean = full_df_clean.append(df)

    full_df = full_df_clean
    full_df_clean = None
#    full_df = full_df.dropna(subset=['x', 'O_Phi', 'fR'])

    fit_df = full_df[full_df['x'] < 20]
    avg_df = full_df.groupby('x').mean().iloc[:20].reset_index(drop=False)

    ar_3_1, r_3_1, pr_3_1, f_r2_3_1 = fit_curve(
        fit_df['x'],
        fit_df['fR'].to_numpy(),
        func_3_param_1
    )
    
    ar_3_2, r_3_2, pr_3_2, f_r2_3_2 = fit_curve(
        fit_df['x'],
        fit_df['fR'].to_numpy(),
        func_3_param_2
    )

    ar_2_1, r_2_1, pr_2_1, f_r2_2_1 = fit_curve(
        fit_df['x'],
        fit_df['fR'].to_numpy(),
        func_2_param_1
    )

    ar_2_2, r_2_2, pr_2_2, f_r2_2_2 = fit_curve(
        fit_df['x'],
        fit_df['fR'].to_numpy(),
        func_2_param_2
    )

    ar_2_3, r_2_3, pr_2_3, f_r2_2_3 = fit_curve(
        fit_df['x'],
        fit_df['fR'].to_numpy(),
        func_2_param_3
    )


    stats = pandas.DataFrame(data={
        'func3_1': [ar_3_1, r_3_1, pr_3_1],
        'func3_2': [ar_3_2, r_3_2, pr_3_2],
        'func2_1': [ar_2_1, r_2_1, pr_2_1],
        'func2_2': [ar_2_2, r_2_2, pr_2_2],
        'func2_3': [ar_2_3, r_2_3, pr_2_3],
    })

    return stats, avg_df, full_df 



def get_mobility_yearly(images, mask):

    A = len(np.where(mask == 1)[1])
    year_range = list(images.keys())
    ranges = [year_range[i:] for i, yr in enumerate(year_range)]
    river_dfs = {}
    for yrange in ranges:
        data = {
            'year': [],
            'i': [],
            'D': [],
            'D/A': [],
            'Phi': [],
            'O_Phi': [],
            'fR': [],
            'fw_b': [],
            'fd_b': [],
        }
        length = images[yrange[0]].shape[0]
        width = images[yrange[0]].shape[1]
        long = len(yrange)
        all_images = np.empty((length, width, long))
        years = []
        for j, year in enumerate(yrange):
            years.append(year)
            im = images[str(year)]
            where = np.where(~mask)
            im[where] = 0
            all_images[:, :, j] = im

        baseline = all_images[:, :, 0]

        w_b = len(np.where(baseline == 1)[0])
        fb = mask - baseline
        fw_b = w_b / A
        fd_b = np.sum(fb) / A
        Na = A * fd_b

        for j in range(all_images.shape[2]):
            im = all_images[:, :, j]

            kb = (
                np.sum(all_images[:,:, :j + 1], axis=(2)) 
                + mask
            )
            kb[np.where(kb != 1)] = 0
            Nb = np.sum(kb)
#            fR = 1 - (Nb / (A * fd_b))
#            fR = (1 - ((Nb / (A * fd_b)))) * (fd_b**6)
#            fR = 1 - (Nb / ((A * fd_b) * fw_b))
#            fR = (Nb / A) * (fd_b**3)
            fR = (Na / w_b) - (Nb / w_b)

            # Calculate D - EQ. (1)
            D = np.sum(np.abs(np.subtract(baseline, im)))

            # Calculate D / A
            D_A = D / A

            # Calculate Phi
            w_t = len(np.where(im == 1)[0])
            fw_t = w_t / A
            fd_t = (A - w_t) / A

            PHI = (fw_b * fd_t) + (fd_b * fw_t)

            # Calculate O_Phi
            O_PHI = 1 - (D / (A * PHI))

            data['i'].append(j)
            data['D'].append(D)
            data['D/A'].append(D_A)
            data['Phi'].append(PHI)
            data['O_Phi'].append(O_PHI)
            data['fR'].append(fR)
            data['fw_b'].append(fw_b)
            data['fd_b'].append(fd_b)
        data['year'] = years
        river_dfs[yrange[0]] = pandas.DataFrame(data=data)

    return river_dfs


def get_images(poly, gif, filt, root):
    out = os.path.join(root, '{}')
    paths = pull_esa(poly, out)

    missing_rivers = []
    for river, items in paths.items():
        if not items:
            missing_rivers.append(river)

    didnt_pull = os.path.join(
        root,
        f'error_rivers.txt'
    )
    if missing_rivers:
        with open(didnt_pull, 'w') as f:
            for river in missing_rivers:
                f.write(f'{river}\n')


    paths = {k: v for k, v in paths.items() if v}

    rivers = []
    for river, path_list in paths.items():
        print(river)

        if not path_list:
            continue

        mask = create_mask_shape(
            poly,
            river,
            path_list
        )
        clean_mask = clean_channel_belt(
            mask, 100
        )

        images, metas = clean_esa(
            poly, 
            river, 
            path_list
        )

        clean_images = filter_images(
            images,
            mask,
            thresh=filt
        )

        river_dfs = get_mobility_yearly(
            clean_images,
            clean_mask,
        )

        full_df = pandas.DataFrame()
        for year, df in river_dfs.items():
            rnge = f"{year}_{df.iloc[-1]['year']}"
            df['dt'] = pandas.to_datetime(
                df['year'],
                format='%Y'
            )
            df['range'] = rnge

            full_df = full_df.append(df)

        out_path = os.path.join(
            out.format(river), 
            f'{river}_yearly_mobility.csv'
        )
        full_df.to_csv(out_path)
        rivers.append(river)

    return rivers




def make_gifs(rivers, root):
    for river in rivers:
        print(river)
        fp = sorted(
            glob.glob(os.path.join(root, f'{river}/*mobility.csv'))
        )[0]
        fp_in = os.path.join(
            root, f'{river}/temps/*.tif'
        )
        fp_out = os.path.join(
            root, f'{river}/{river}_cumulative.gif'
        )
        stat_out = os.path.join(
            root, f'{river}/{river}_mobility_stats.csv'
        )

#        yield make_gif(fp, fp_in, fp_out, stat_out)
        yield test_equations(fp, fp_in, fp_out, stat_out)


def func_3_param_1(x, a, m, p):
    return ((a - p) * np.exp(-m * x)) + p

def func_3_param_2(x, a, m, p):
    return (a * np.exp(-m * x)) + p

def func_2_param_1(x, a, m, p):
    return (a * np.exp(-m * x))

def func_2_param_2(x, a, m, p):
    return (a * np.exp(m * x))

def func_2_param_3(x, a, m, p):
    return (a * np.exp(m * x)) - a


def fit_curve(x, y, fun):
    # Fitting
    popt, pcov = curve_fit(fun, x, y, p0=[1, .00001, 0], maxfev=10000)
    # R-squared
    residuals = y - fun(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return (*popt), r_squared


if __name__ == '__main__':

    poly = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Development/PowderSizeTest.gpkg'
    out = '/Users/greenberg/Code/Github/mobility/dev/test/'

    rivers = get_images(poly, 'true', .0001, out)

    stat_list = []
    dfs = []
    fulls = []
    stats = make_gifs(rivers, out)
    for stat, avg_df, full_df in stats:
        stat_list.append(stat)
        dfs.append(avg_df)
        fulls.append(full_df)
        print(stat)

funcs = [
    func_3_param_1,
    func_3_param_2,
    func_2_param_1,
    func_2_param_2,
    func_2_param_3,
]
x = np.linspace(0, 30)
fig, axs = plt.subplots(1, 4, sharex=True, sharey=True)
for i, df in enumerate(dfs):
    stats = stat_list[i]
    full_df = fulls[i]
    for j, column in enumerate(stats.columns):
        pred = funcs[j](
            x,
            stats[column].iloc[0],
            stats[column].iloc[1],
            stats[column].iloc[2],
        )
        axs[i].plot(x, pred, label=column)

    axs[i].scatter(df['i'], df['fR'], color='black', zorder=5)
    axs[i].scatter(
        full_df['i'], full_df['fR'], 
        facecolor='white', edgecolor='black',
        s=30, zorder=0
    )

axs[0].set_title('Tight Window')
axs[1].set_title('Channel-belt Window')
axs[2].set_title('Wide Window')

axs[0].set_ylabel('fR')
axs[1].set_xlabel('Year from T0')
plt.legend()
plt.show()
