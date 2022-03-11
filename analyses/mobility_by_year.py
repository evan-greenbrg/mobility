import io
import os
import glob
import pandas
from scipy.optimize import curve_fit
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt


def func_3_param(x, a, m, p):
    return ((a - p) * np.exp(-m * x)) + p

def func_2_param(x, a, m, p):
    return ((a) * np.exp(-m * x))


def fit_curve(x, y, fun):
    # Fitting
    popt, pcov = curve_fit(fun, x, y, p0=[1, .00001, 0], maxfev=10000)
    # R-squared
    residuals = y - fun(x, *popt)
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y - np.mean(y))**2)
    r_squared = 1 - (ss_res / ss_tot)

    return (*popt), r_squared


def get_stats(df, fun):
    # Make x column
    df['x'] = df['year'] - df.iloc[0]['year']

    # Fit curves to normalized overlap
    am, m, pm, o_r2 = fit_curve(
        df['x'],
        df['O_Phi'].to_numpy(),
        fun
    )

    # Rework fraction
    ar, r, pr, f_r2 = fit_curve(
        df['x'],
        1 - df['fR'].to_numpy(),
        fun
    )

    return am, m, pm, ar, r, pr


def fit_regression(x, y):
    x = x.to_numpy()
    y = y.to_numpy()
    xgrid = np.array([i for i in np.arange(x.min(), x.max(), .01)])
    xreg = sm.add_constant(x)
    model = sm.OLS(y, xreg)
    results = model.fit()
    params = results.params
    r2 = results.rsquared

    return params[1], params[0], r2


root = '/Users/Evan/Documents/Mobility/GIS/Development'
fps = glob.glob(os.path.join(root, '*', '*mobility.csv'))
fp = fps[1]

full_df = pandas.read_csv(fp).groupby('range')
data = {
    'year': [],
    'am_3': [],
    'm_3': [],
    'pm_3': [],
    'ar_3': [],
    'r_3': [],
    'pr_3': [],
    'am_2': [],
    'm_2': [],
    'pm_2': [],
    'ar_2': [],
    'r_2': [],
    'pr_2': [],
}
for year, df in full_df:
    year = year.split('_')[0]
    data['year'].append(year)
    if len(df) > 2:
        am_3, m_3, pm_3, ar_3, r_3, pr_3 = get_stats(df, func_3_param)
        am_2, m_2, pm_2, ar_2, r_2, pr_2 = get_stats(df, func_2_param)
    else:
        am_3 = m_3 = pm_3 = ar_3 = r_3 = pr_3 = None
        am_2 = m_2 = pm_2 = ar_2 = r_2 = pr_3 = None

    data['am_3'].append(am_3)
    data['m_3'].append(m_3)
    data['pm_3'].append(pm_3)
    data['ar_3'].append(ar_3)
    data['r_3'].append(r_3)
    data['pr_3'].append(pr_3)
    data['am_2'].append(am_2)
    data['m_2'].append(m_2)
    data['pm_2'].append(pm_2)
    data['ar_2'].append(ar_2)
    data['r_2'].append(r_2)
    data['pr_2'].append(pr_2)

stat_df = pandas.DataFrame(data=data)
stat_df['year'] = stat_df['year'].astype(int)


xcol = 'm_3'
a = 'am_3'
m = 'm_3'
p = 'pm_3'
scol = 'O_Phi'
fun = func_3_param
images = []
fp_root = '/Users/Evan/Documents/Mobility/GIS/Development/MultiYearTaiwan1/'
fp_name = 'MultiYearTaiwan1_timeseries_3_param_m.gif'
fp_out = os.path.join(fp_root, fp_name)
for (i, row), (year, df) in zip(stat_df.iterrows(), full_df):
    if i >= len(stat_df) - 2:
        continue
    img_buf = io.BytesIO()
    df = df.reset_index(drop=True)
    fig, axs = plt.subplots(2,1, constrained_layout=True, figsize=(10, 7))

    # Plot the data
    axs[0].plot(
        stat_df['year'], 
        stat_df[xcol],
        color='black',
        zorder=1
    )
    axs[0].scatter(
        row['year'], 
        row[xcol], 
        s=200, 
        facecolor='red', 
        edgecolor='black',
        zorder=2
    )
    axs[1].plot(
        [i for i in range(0, 30)], 
        fun(
            np.array([i for i in range(0, 30)]),
            row[a],
            row[m],
            row[p]
        ),
        color='black',
        zorder=1
    )
    if scol == 'O_Phi':
        y = df[scol]
    elif scol == 'fR':
        y = 1 - df[scol]
    axs[1].scatter(
        df.index,
        y,
        s=200,
        facecolor='white',
        edgecolor='black'
    )

    # Some text
    axs[0].text(
        row['year'],
        row[xcol] + .005,
        f'Exponent: {round(row[m], 2)}',
        horizontalalignment='left', 
    )
    r2 = r2_score(
        df[scol],
        fun(
            df.index,
            row[a],
            row[m],
            row[p]
        )
    )
    axs[1].text(
        .85,
        .95,
        f'R2: {round(r2, 2)}',
        horizontalalignment='left', 
        transform=axs[1].transAxes
    )

    # Labels
    axs[0].set_xlabel('Year')
    axs[0].set_ylabel(xcol)
    axs[1].set_xlabel('Years from t0')
    axs[1].set_ylabel(scol)

    plt.savefig(img_buf, format='png')
    plt.close('all')
    images.append(Image.open(img_buf))

img, *imgs = images 
img.save(
    fp=fp_out, 
    format='GIF', 
    append_images=imgs,         
    save_all=True, 
    duration=400, 
    loop=100
)

