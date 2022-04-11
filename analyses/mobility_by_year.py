import random
import io
import os
from PIL import Image
import glob
import pandas
from scipy.optimize import curve_fit
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt
from sklearn.metrics import r2_score


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

river = 'Taiwan1'
root = f'/Users/greenberg/Documents/PHD/Projects/Mobility/TaiwanAnalysis/{river}'
fps = glob.glob(os.path.join(root, '*', '*mobility.csv'))
stat_dfs = pandas.DataFrame()
full_dfs = {} 
for fp in fps:
    section = fp.split('/')[-2]

    full_df = pandas.read_csv(fp)
    full_df['section'] = section
    full_df = full_df.groupby('range')

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
    stat_df['section'] = section

    stat_dfs = stat_dfs.append(stat_df)
    full_dfs[section] = full_df 


xcol = 'm_3'
a = 'am_3'
m = 'm_3'
p = 'pm_3'
# scol = 'fR'
scol = 'O_Phi'
fun = func_3_param
images = []
fp_root = f'/Users/greenberg/Documents/PHD/Projects/Mobility/TaiwanAnalysis/{river}'
fp_name = 'Combined_cumulative_3_param_r.gif'
fp_out = os.path.join(fp_root, fp_name)

med = stat_dfs.groupby('year').median()
lo = stat_dfs.groupby('year').quantile(.25)
up = stat_dfs.groupby('year').quantile(.75)
for section in stat_dfs['section'].unique():
    temp = stat_dfs[stat_dfs['section'] == section]
    plt.scatter(
        temp['year'],
        temp['m_3']
    )
fig, axs = plt.subplots(3,1, constrained_layout=True, figsize=(10, 7))
axs[0].plot(
    med.index,
    med['m_3'],
    zorder=5,
    color='red'
)
axs[0].fill_between(
    med.index,
    lo['m_3'],
    up['m_3'],
    zorder=-1,
    color='lightgray',
)

axs[1].plot(
    med.index,
    med['r_2'],
    zorder=5,
    color='red'
)
axs[1].fill_between(
    med.index,
    lo['r_2'],
    up['r_2'],
    zorder=-1,
    color='lightgray',
)
axs[2].plot(
    med.index,
    med['m_3'] / med['r_2'],
    zorder=5,
    color='red'
)
axs[2].fill_between(
    med.index,
    lo['m_3'] / up['r_2'],
    up['m_3'] / lo['r_2'],
    zorder=-1,
    color='lightgray',
)
plt.show()
