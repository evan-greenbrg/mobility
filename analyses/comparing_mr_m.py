import os
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


def main_plot(root, river_name):

    path = os.path.join(root, '{}*/*mobility.csv')
    fps = glob.glob(path.format(river_name))

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
        'am2': [],
        'm2': [],
        'pm2': [],
        'o_r2_2': [],
        'ar2': [],
        'r2': [],
        'pr2': [],
        'f_r2_2': [],
        'av_am': [],
        'av_m': [],
        'av_pm': [],
        'av_o_r2': [],
        'av_ar': [],
        'av_r': [],
        'av_pr': [],
        'av_f_r2': [],
        'av_am2': [],
        'av_m2': [],
        'av_pm2': [],
        'av_o_r2_2': [],
        'av_ar2': [],
        'av_r2': [],
        'av_pr2': [],
        'av_f_r2_2': [],
    }
    fig, axs = plt.subplots(2, 4, figsize=(12,6))
    for fp in fps:
        print(fp)
        name = fp.split('/')[-2]
        full_df = pandas.read_csv(fp)
        full_df['O_Phi'] = full_df['O_Phi'].replace(0, .000001)
        full_df['fR'] = full_df['fR'].replace(0, .000001)

        full_df_clean = pandas.DataFrame()
        for group, df in full_df.groupby('range'):
            df['x'] = df['year'] - df.iloc[0]['year']
            full_df_clean = full_df_clean.append(df)

        full_df = full_df_clean
        full_df_clean = None

        full_df = full_df.dropna(subset=['O_Phi', 'fR'])

        avg_df = full_df.groupby('x').mean().reset_index(drop=False)

        # Fit curves to normalized overlap
        am, m, pm, o_r2 = fit_curve(
            full_df['x'],
            full_df['O_Phi'].to_numpy(),
            func
        )

        ar, r, pr, f_r2 = fit_curve(
            full_df['x'],
            1 - full_df['fR'].to_numpy(),
            func
        )

        am2, m2, pm2, o_r2_2 = fit_curve(
            full_df['x'],
            full_df['O_Phi'].to_numpy(),
            func2
        )

        ar2, r2, pr2, f_r2_2 = fit_curve(
            full_df['x'],
            1 - full_df['fR'].to_numpy(),
            func2
        )

        # Fit avg curves to normalized overlap
        av_am, av_m, av_pm, av_o_r2 = fit_curve(
            avg_df['x'],
            avg_df['O_Phi'].to_numpy(),
            func
        )

        av_ar, av_r, av_pr, av_f_r2 = fit_curve(
            avg_df['x'],
            1 - avg_df['fR'].to_numpy(),
            func
        )

        av_am2, av_m2, av_pm2, av_o_r2_2 = fit_curve(
            avg_df['x'],
            avg_df['O_Phi'].to_numpy(),
            func2
        )

        av_ar2, av_r2, av_pr2, av_f_r2_2 = fit_curve(
            avg_df['x'],
            1 - avg_df['fR'].to_numpy(),
            func2
        )

        xgrid = np.array([i for i in range(0, 30)])
        ypred_m = func(xgrid, am, m, pm)
        ypred_r = func(xgrid, ar, r, pr)
        ypred_m2 = func2(xgrid, am2, m2, pm2)
        ypred_r2 = func2(xgrid, ar2, r2, pr2)
        av_ypred_m = func(xgrid, av_am, av_m, av_pm)
        av_ypred_r = func(xgrid, av_ar, av_r, av_pr)
        av_ypred_m2 = func2(xgrid, av_am2, av_m2, av_pm2)
        av_ypred_r2 = func2(xgrid, av_ar2, av_r2, av_pr2)

        axs[0, 0].plot(xgrid, ypred_m)
        axs[0, 0].scatter(
            full_df['x'], 
            full_df['O_Phi'], 
            label=name,
            facecolor='white',
            edgecolor='black',
            s=10
        )
        axs[0, 0].plot(
            avg_df['x'],
            avg_df['O_Phi'],
            color='black'
        )
        axs[0, 0].scatter(
            avg_df['x'],
            avg_df['O_Phi'],
            color='black',
            s=35
        )


        axs[1, 0].plot(xgrid, ypred_r)
        axs[1, 0].scatter(
            full_df['x'], 
            1 - full_df['fR'], 
            label=name,
            facecolor='white',
            edgecolor='black',
            s=10
        )
        axs[1, 0].plot(
            avg_df['x'],
            1 - avg_df['fR'],
            color='black'
        )
        axs[1, 0].scatter(
            avg_df['x'],
            1 - avg_df['fR'],
            color='black',
            s=35
        )

        axs[0, 1].plot(xgrid, ypred_m2)
        axs[0, 1].scatter(
            full_df['x'], 
            full_df['O_Phi'], 
            label=name,
            facecolor='white',
            edgecolor='black',
            s=10
        )
        axs[0, 1].plot(
            avg_df['x'],
            avg_df['O_Phi'],
            color='black'
        )
        axs[0, 1].scatter(
            avg_df['x'],
            avg_df['O_Phi'],
            color='black',
            s=35
        )

        axs[1, 1].plot(xgrid, ypred_r2)
        axs[1, 1].scatter(
            full_df['x'], 
            1 - full_df['fR'], 
            label=name,
            facecolor='white',
            edgecolor='black',
            s=10
        )
        axs[1, 1].plot(
            avg_df['x'],
            1 - avg_df['fR'],
            color='black'
        )
        axs[1, 1].scatter(
            avg_df['x'],
            1 - avg_df['fR'],
            color='black',
            s=35
        )

        axs[0, 2].plot(xgrid, av_ypred_m)
        axs[0, 2].scatter(
            full_df['x'], 
            full_df['O_Phi'], 
            label=name,
            facecolor='white',
            edgecolor='black',
            s=10
        )
        axs[0, 2].plot(
            avg_df['x'],
            avg_df['O_Phi'],
            color='black'
        )
        axs[0, 2].scatter(
            avg_df['x'],
            avg_df['O_Phi'],
            color='black',
            s=35
        )


        axs[1, 2].plot(xgrid, av_ypred_r)
        axs[1, 2].scatter(
            full_df['x'], 
            1 - full_df['fR'], 
            label=name,
            facecolor='white',
            edgecolor='black',
            s=10
        )
        axs[1, 2].plot(
            avg_df['x'],
            1 - avg_df['fR'],
            color='black'
        )
        axs[1, 2].scatter(
            avg_df['x'],
            1 - avg_df['fR'],
            color='black',
            s=35
        )

        axs[0, 3].plot(xgrid, av_ypred_m2)
        axs[0, 3].scatter(
            full_df['x'], 
            full_df['O_Phi'], 
            label=name,
            facecolor='white',
            edgecolor='black',
            s=10
        )
        axs[0, 3].plot(
            avg_df['x'],
            avg_df['O_Phi'],
            color='black'
        )
        axs[0, 3].scatter(
            avg_df['x'],
            avg_df['O_Phi'],
            color='black',
            s=35
        )

        axs[1, 3].plot(xgrid, av_ypred_r2)
        axs[1, 3].scatter(
            full_df['x'], 
            1 - full_df['fR'], 
            label=name,
            facecolor='white',
            edgecolor='black',
            s=10
        )
        axs[1, 3].plot(
            avg_df['x'],
            1 - avg_df['fR'],
            color='black'
        )
        axs[1, 3].scatter(
            avg_df['x'],
            1 - avg_df['fR'],
            color='black',
            s=35
        )

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
        data['am2'].append(am2)
        data['m2'].append(m2)
        data['pm2'].append(pm2)
        data['o_r2_2'].append(o_r2_2)
        data['ar2'].append(ar2)
        data['r2'].append(r2)
        data['pr2'].append(pr2)
        data['f_r2_2'].append(f_r2_2)

        data['av_am'].append(av_am)
        data['av_m'].append(av_m)
        data['av_pm'].append(av_pm)
        data['av_o_r2'].append(av_o_r2)
        data['av_ar'].append(av_ar)
        data['av_r'].append(av_r)
        data['av_pr'].append(av_pr)
        data['av_f_r2'].append(av_f_r2)
        data['av_am2'].append(av_am2)
        data['av_m2'].append(av_m2)
        data['av_pm2'].append(av_pm2)
        data['av_o_r2_2'].append(av_o_r2_2)
        data['av_ar2'].append(av_ar2)
        data['av_r2'].append(av_r2)
        data['av_pr2'].append(av_pr2)
        data['av_f_r2_2'].append(av_f_r2_2)
    river_df = pandas.DataFrame(data=data)
    axs[0,0].text(
        0, 
        1.05,
        f"3-Param M: {round(river_df['m'].mean(), 5)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[0,0].transAxes
    )
    axs[0,0].text(
        .5, 
        .95,
        f"R2: {round(river_df['o_r2'].mean(), 3)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[0,0].transAxes
    )

    axs[0,1].text(
        0, 
        1.05,
        f"2-Param M: {round(river_df['m2'].mean(), 5)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[0,1].transAxes
    )
    axs[0,1].text(
        .5, 
        .95,
        f"R2: {round(river_df['o_r2_2'].mean(), 3)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[0,1].transAxes
    )

    axs[1,0].text(
        0, 
        1.05,
        f"3-Param R: {round(river_df['r'].mean(), 5)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[1,0].transAxes
    )
    axs[1,0].text(
        .5, 
        .95,
        f"R2: {round(river_df['f_r2'].mean(), 3)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[1,0].transAxes
    )

    axs[1,1].text(
        0, 
        1.05,
        f"2-Param R: {round(river_df['r2'].mean(), 5)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[1,1].transAxes
    )
    axs[1,1].text(
        .5, 
        .95,
        f"R2: {round(river_df['f_r2_2'].mean(), 3)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[1,1].transAxes
    )

    axs[0,2].text(
        0, 
        1.05,
        f"Avg. 3-Param M: {round(river_df['av_m'].mean(), 5)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[0,2].transAxes
    )
    axs[0,2].text(
        .5, 
        .95,
        f"R2: {round(river_df['av_o_r2'].mean(), 3)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[0,2].transAxes
    )

    axs[0,3].text(
        0, 
        1.05,
        f"Avg. 2-Param M: {round(river_df['av_m2'].mean(), 5)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[0,3].transAxes
    )
    axs[0,3].text(
        .5, 
        .95,
        f"R2: {round(river_df['av_o_r2_2'].mean(), 3)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[0,3].transAxes
    )

    axs[1,2].text(
        0, 
        1.05,
        f"Avg. 3-Param R: {round(river_df['av_r'].mean(), 5)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[1,2].transAxes
    )
    axs[1,2].text(
        .5, 
        .95,
        f"R2: {round(river_df['av_f_r2'].mean(), 3)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[1,2].transAxes
    )

    axs[1,3].text(
        0, 
        1.05,
        f"Avg. 2-Param R: {round(river_df['av_r2'].mean(), 5)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[1,3].transAxes
        )
    axs[1,3].text(
        .5, 
        .95,
        f"R2: {round(river_df['av_f_r2_2'].mean(), 3)}",
        horizontalalignment='left', 
        verticalalignment='center', 
        transform=axs[1,3].transAxes
    )

    fig.suptitle(river_name)
    plt.savefig(
        os.path.join(root, f'{river_name}.png'),
        format='png'
    )
    plt.close('all')
#    plt.show()


root = '/Users/greenberg/Documents/PHD/Projects/Mobility/GIS/Comparing/big_compare/box_based_channel_belt'

rivers = [
    'Sabine',
    'Red_River_Downstream',
    'Brazos_River',
    'Trinity_River',
    'Willamette',
    'Rio_Coco',
    'Amazon_River',
    'Rio_Solimoes',
    'Rio_Jurua',
    'Rio_Purus',
    'Rio_Madre_de_Dios',
    'Rio_Beni',
    'Rio_Mamore',
    'Aquapei_Reach_B',
    'Rio_Xingu',
    'Niger_River',
    'Chad',
    'Indus',
    'Ramganga_River',
    'Khowai_River',
    'Kazak2',
    'Kazak1',
    'Uda',
]
for river in rivers:
    main_plot(root, river)
