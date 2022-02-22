import glob
import pandas
from scipy.optimize import curve_fit
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt


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


def get_stats(df):
    # Make x column
    df['x'] = df['year'] - df.iloc[0]['year']

    # Fit curves to normalized overlap
    am, m, pm, o_r2 = fit_curve(
        df['x'],
        df['O_Phi'].to_numpy()
    )

    # Rework fraction
    ar, r, pr, f_r2 = fit_curve(
        df['x'],
        1 - df['fR'].to_numpy()
    )

    # Instantaneous mobility
    zeta_median = df['zeta'].median()

    return m, r, zeta_median


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


ignore = [
    'Indus',
    'Minnesota_River_Downstream',
    'Okavango_River',
    'Pearl',
    'Strickland_River',
    'Tarim',
    'Wabash_River',
    'Escambia',
    'White_River',
    'Yellow_River'
]

fps = glob.glob('/home/greenberg/ExtraSpace/PhD/Projects/Mobility/GIS/Comparing/Rivers/*/*mobility.csv')
data = {
    'River': [],
    'M': [],
    'R': [],
    'Zeta': [],
}
for fp in fps:
    river = fp.split('/')[-2]
    if river in ignore:
        continue
    df = pandas.read_csv(fp)
    m, r, zeta = get_stats(df)

    data['River'].append(river)
    data['M'].append(m)
    data['R'].append(r)
    data['Zeta'].append(zeta)

df = pandas.DataFrame(data=data)

# See if I can join
full_fp = '/home/greenberg/ExtraSpace/PhD/Projects/Mobility/GIS/Comparing/FullData021822.csv'
full = pandas.read_csv(full_fp)

df = df.merge(full, on='River')
x = df['Mr*']
y = df['M']

fig, axs = plt.subplots(3, 3)
axs[0,0].scatter(df['Mr [m/yr]'], df['M'])
axs[0,1].scatter(df['Mr [m/yr]'], df['R'])
axs[0,2].scatter(df['Mr [m/yr]'], df['Zeta'])

axs[1,0].scatter(df['Width [m]'], df['M'])
axs[1,1].scatter(df['Width [m]'], df['R'])
axs[1,2].scatter(df['Width [m]'], df['Zeta'])

axs[2,0].scatter(df['Mr*'], df['M'])
axs[2,1].scatter(df['Mr*'], df['R'])
axs[2,2].scatter(df['Mr*'], df['Zeta'])

plt.show()

