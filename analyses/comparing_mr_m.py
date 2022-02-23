import glob
import pandas
from scipy.optimize import curve_fit
import numpy as np
import statsmodels.api as sm
from matplotlib import pyplot as plt


def func(x, a, m, p):
    return ((a - p) * np.exp(-m * x)) + p

def func2(x, a, m, p):
    return ((a) * np.exp(-m * x))


def fit_curve(x, y, fun):
    # Fitting
    popt, pcov = curve_fit(fun, x, y, p0=[1, .00001, 0], maxfev=10000)
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
    fw_b = df['fw_b'].median()
    o_phi = df['O_Phi'].median()
    phi = df['Phi'].median()
    fr = df['fR'].median()
    fd_b = df['fd_b'].median()

    tao_ch = (fw_b * (1 - (phi / 2)) * (1 - o_phi)) / zeta_median
    tao_f = (fd_b * fr) / zeta_median

    return m, r, zeta_median, tao_ch, tao_f


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


root = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Comparing/{}/*/*mobility.csv'
fps = glob.glob(root.format('Beni'))

data = {
    'Name': [],
    'am': [],
    'm': [],
    'pm': [],
    'o_r2': [],
    'ar': [],
    'r': [],
    'pr': [],
    'f_r2': [],
}
fig, axs = plt.subplots(2, 2)
for fp in fps:
    name = fp.split('/')[-2]
    df = pandas.read_csv(fp)

    df['x'] = df['year'] - df.iloc[0]['year']

    # Fit curves to normalized overlap
    am, m, pm, o_r2 = fit_curve(
        df['x'],
        df['O_Phi'].to_numpy(),
        func
    )

    # Rework fraction
    ar, r, pr, f_r2 = fit_curve(
        df['x'],
        1 - df['fR'].to_numpy(),
        func
    )

    am2, m2, pm2, o_r2_2 = fit_curve(
        df['x'],
        df['O_Phi'].to_numpy(),
        func2
    )

    # Rework fraction
    ar2, r2, pr2, f_r2_2 = fit_curve(
        df['x'],
        1 - df['fR'].to_numpy(),
        func2
    )

    xgrid = np.array([i for i in range(0, 30)])
    ypred_m = func(xgrid, am, m, pm)
    ypred_r = func(xgrid, ar, r, pr)
    axs[0, 0].plot(xgrid, ypred_m)
    axs[0, 0].scatter(df['x'], df['O_Phi'], label=name)
    axs[1, 0].plot(xgrid, ypred_r)
    axs[1, 0].scatter(df['x'], 1 - df['fR'], label=name)

    ypred_m2 = func2(xgrid, am2, m2, pm2)
    ypred_r2 = func2(xgrid, ar2, r2, pr2)
    axs[0, 1].plot(xgrid, ypred_m2)
    axs[0, 1].scatter(df['x'], df['O_Phi'], label=name)
    axs[1, 1].plot(xgrid, ypred_r2)
    axs[1, 1].scatter(df['x'], 1 - df['fR'], label=name)
    axs[0,0].set_ylabel('Ophi')
    axs[1,0].set_ylabel('1 - fR')
    axs[1,1].set_xlabel('year')
    axs[1,0].set_xlabel('year')

    data['Name'].append(name)
    data['am'].append(am)
    data['m'].append(m)
    data['pm'].append(pm)
    data['o_r2'].append(o_r2)
    data['ar'].append(ar)
    data['r'].append(r)
    data['pr'].append(pr)
    data['f_r2'].append(f_r2)
beni_df = pandas.DataFrame(data=data)
plt.legend()
plt.show()

root = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Comparing/{}/*/*mobility.csv'
fps = glob.glob(root.format('Escambia'))

data = {
    'Name': [],
    'am': [],
    'm': [],
    'pm': [],
    'o_r2': [],
    'ar': [],
    'r': [],
    'pr': [],
    'f_r2': [],
}
fig, axs = plt.subplots(2, 2)
for fp in fps:
    name = fp.split('/')[-2]
    df = pandas.read_csv(fp)

    df['x'] = df['year'] - df.iloc[0]['year']

    # Fit curves to normalized overlap
    am, m, pm, o_r2 = fit_curve(
        df['x'],
        df['O_Phi'].to_numpy(),
        func
    )

    # Rework fraction
    ar, r, pr, f_r2 = fit_curve(
        df['x'],
        1 - df['fR'].to_numpy(),
        func
    )

    am2, m2, pm2, o_r2_2 = fit_curve(
        df['x'],
        df['O_Phi'].to_numpy(),
        func2
    )

    # Rework fraction
    ar2, r2, pr2, f_r2_2 = fit_curve(
        df['x'],
        1 - df['fR'].to_numpy(),
        func2
    )

    xgrid = np.array([i for i in range(0, 30)])
    ypred_m = func(xgrid, am, m, pm)
    ypred_r = func(xgrid, ar, r, pr)
    axs[0, 0].plot(xgrid, ypred_m)
    axs[0, 0].scatter(df['x'], df['O_Phi'], label=name)
    axs[1, 0].plot(xgrid, ypred_r)
    axs[1, 0].scatter(df['x'], 1 - df['fR'], label=name)

    ypred_m2 = func2(xgrid, am2, m2, pm2)
    ypred_r2 = func2(xgrid, ar2, r2, pr2)
    axs[0, 1].plot(xgrid, ypred_m2)
    axs[0, 1].scatter(df['x'], df['O_Phi'], label=name)
    axs[1, 1].plot(xgrid, ypred_r2)
    axs[1, 1].scatter(df['x'], 1 - df['fR'], label=name)
    axs[0,0].set_ylabel('Ophi')
    axs[1,0].set_ylabel('1 - fR')
    axs[1,1].set_xlabel('year')
    axs[1,0].set_xlabel('year')

    data['Name'].append(name)
    data['am'].append(am)
    data['m'].append(m)
    data['pm'].append(pm)
    data['o_r2'].append(o_r2)
    data['ar'].append(ar)
    data['r'].append(r)
    data['pr'].append(pr)
    data['f_r2'].append(f_r2)

escambia_df = pandas.DataFrame(data=data)
plt.legend()
plt.show()

root = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Comparing/{}/*/*mobility.csv'
fps = glob.glob(root.format('Watut'))

data = {
    'Name': [],
    'am': [],
    'm': [],
    'pm': [],
    'o_r2': [],
    'ar': [],
    'r': [],
    'pr': [],
    'f_r2': [],
}
fig, axs = plt.subplots(2, 2)
for fp in fps:
    name = fp.split('/')[-2]
    df = pandas.read_csv(fp)

    df['x'] = df['year'] - df.iloc[0]['year']

    # Fit curves to normalized overlap
    am, m, pm, o_r2 = fit_curve(
        df['x'],
        df['O_Phi'].to_numpy(),
        func
    )

    # Rework fraction
    ar, r, pr, f_r2 = fit_curve(
        df['x'],
        1 - df['fR'].to_numpy(),
        func
    )

    am2, m2, pm2, o_r2_2 = fit_curve(
        df['x'],
        df['O_Phi'].to_numpy(),
        func2
    )

    # Rework fraction
    ar2, r2, pr2, f_r2_2 = fit_curve(
        df['x'],
        1 - df['fR'].to_numpy(),
        func2
    )

    xgrid = np.array([i for i in range(0, 30)])
    ypred_m = func(xgrid, am, m, pm)
    ypred_r = func(xgrid, ar, r, pr)
    axs[0, 0].plot(xgrid, ypred_m)
    axs[0, 0].scatter(df['x'], df['O_Phi'], label=name)
    axs[1, 0].plot(xgrid, ypred_r)
    axs[1, 0].scatter(df['x'], 1 - df['fR'], label=name)

    ypred_m2 = func2(xgrid, am2, m2, pm2)
    ypred_r2 = func2(xgrid, ar2, r2, pr2)
    axs[0, 1].plot(xgrid, ypred_m2)
    axs[0, 1].scatter(df['x'], df['O_Phi'], label=name)
    axs[1, 1].plot(xgrid, ypred_r2)
    axs[1, 1].scatter(df['x'], 1 - df['fR'], label=name)

    data['Name'].append(name)
    data['am'].append(am)
    data['m'].append(m)
    data['pm'].append(pm)
    data['o_r2'].append(o_r2)
    data['ar'].append(ar)
    data['r'].append(r)
    data['pr'].append(pr)
    data['f_r2'].append(f_r2)

watut_df = pandas.DataFrame(data=data)
plt.legend()
plt.show()


xgrid = np.array([i for i in range(0, 30)])
for i, row in beni_df.iterrows():
    ypred = func(xgrid, row['am'], row['m'], row['pm'])
    print(ypred)
    plt.plot(xgrid, ypred, color='blue')
for i, row in escambia_df.iterrows():
    ypred = func(xgrid, row['am'], row['m'], row['pm'])
    print(ypred)
    plt.plot(xgrid, ypred, color='red')
for i, row in watut_df.iterrows():
    ypred = func(xgrid, row['am'], row['m'], row['pm'])
    print(ypred)
    plt.plot(xgrid, ypred, color='green')
plt.show()


beni_mean = beni_df.mean()
escambia_mean = escambia_df.mean()
watut_mean = watut_df.mean()

beni_r = beni_mean['r']
escambia_r = escambia_mean['r']
watut_r = watut_mean['r']

beni_m = beni_mean['m']
escambia_m = escambia_mean['m']
watut_m = watut_mean['m']

escambia_data = full[full['River'] == 'Escambia']
beni_data = full[full['River'] == 'Rio_Beni']
watut_data = full[full['River'] == 'Watut_River']

fig, axs = plt.subplots(2,1)
xgrid = np.arange(0, .2, .01)
x = [escambia_data['Mr*'], beni_data['Mr*'], watut_data['Mr*']]
y = [escambia_m, beni_m, watut_m]
axs[0].scatter(x, y, color='black')
axs[0].plot(xgrid, xgrid, color='black')
axs[0].set_ylabel('M')
axs[0].set_xlabel('Mr*')

y = [escambia_r, beni_r, watut_r]
axs[1].scatter(x, y, color='black')
axs[1].plot(xgrid, xgrid, color='black')
axs[1].set_ylabel('R')
axs[1].set_xlabel('Mr*')
plt.show()

xgrid = np.array([i for i in range(0, 30)])
for i, row in watut_df.iterrows():
    ypred = func(xgrid, row['am'], row['m'], row['pm'])
    print(ypred)
    plt.plot(xgrid, ypred, label=row['Name'])
plt.legend()
plt.show()
