import pandas
from scipy.optimize import curve_fit
import numpy as np
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


fp = '/home/greenberg/ExtraSpace/PhD/Projects/Mobility/GIS/Development/Himalaya1/Himalaya1_yearly_mobility.csv'
df = pandas.read_csv(fp)
df['x'] = df['year'] - df.iloc[0]['year']


am, m, pm, o_r2 = fit_curve(
    df['x'],
    df['O_Phi'].to_numpy()
)

ar, r, pr, f_r2 = fit_curve(
    df['x'],
    1 - df['fR'].to_numpy()
)

o_pred = func(df['x'], am, m, pm)
f_pred = func(df['x'], ar, r, pr)

fig, axs = plt.subplots(2, 1)
axs[0].scatter(
    df['year'],
    df['O_Phi']
)
axs[0].plot(df['year'], o_pred)

axs[1].scatter(
    df['year'],
    1 - df['fR']
)
axs[1].plot(df['year'], f_pred)
plt.show()
