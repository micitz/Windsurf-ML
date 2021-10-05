"""
Analyze results from Windsurf simulations

Michael Itzkin, 3/24/2021
"""

from Functions import WindsurfResult as ws
from geographiclib.geodesic import Geodesic
from collections import Counter

import scipy.constants as consts
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os


# General settings
font = {'fontname': 'Arial', 'fontweight': 'normal', 'fontsize': 14}
dpi = 300
inches = 3.8
TEXT_DIR = os.path.join('..', '..', 'Text Files')
DATA_DIR = os.path.join('..', '..', 'Data')
FIELD_DIR = os.path.join('..', '..', '..', 'Bogue Banks Field Data')
FIGURE_DIR = os.path.join('..', '..', 'Figures')
WS_DIR = os.path.join('..', '..', '..', 'Windsurf')


"""
Functions to load and format data
"""


def change_df(mdf, fdf, param, field_param, years, profiles):
    """

    :return:
    """

    profile_list, types, results = [], [], []
    for pp in profiles:

        # Do everything once for field data
        profile_list.append(pp)
        types.append('Field')
        results.append(fdf[field_param].loc[(fdf['Year'] == years[-1]) & (fdf['Profile'] == pp)].values[0] - \
                       fdf[field_param].loc[(fdf['Year'] == years[0]) & (fdf['Profile'] == pp)].values[0])

        # Do it all again for model data
        profile_list.append(pp)
        types.append('Model')
        results.append(mdf[param].loc[mdf['Profile'] == pp].iloc[-1] - mdf[param].loc[mdf['Profile'] == pp].iloc[0])

    # Make the DataFrame
    df = pd.DataFrame.from_dict({'Profile': profile_list, 'Type': types, 'Results': results})

    return df


def irribarren(res):
    """
    Calculate the Irribarren number at each time step for the simulation
    and return the time series of values

    res: Windsurf results object
    """

    # Make a DataFrame with the key elemnts from the
    # results object needed to calculate the Irribarren
    # number. This will speed up the calculations
    df = pd.DataFrame()
    df['Beach Slope'] = res.morpho['Beach Slope']
    df['H'] = res.enviro['Hs']
    df['Tp'] = res.enviro['Tp']

    # Calculate the wavelength
    df['L0'] = (consts.g * (df['Tp']**2)) / (2 * consts.pi)

    # Calculate the Iribarren numbers
    df['Iribarren'] = np.tan(df['Beach Slope'].astype(float)) / np.sqrt(df['H'] / df['L0'])

    return df


def params_df():
    """
    Make a dataframe with the:
    - Model morphometric names
    - Field morphometrics names
    - Axis labels
    - Axis limits
    - Unit
    - Error bars
    """

    # Set lists to initialize the DataFrame
    mm = ['Y Crest', 'Y Toe', 'Dune Width', 'Dune Volume', 'X Toe', 'Beach Slope', 'Beach Width', 'Beach Volume', 'X MHW']
    fm = ['yCrest', 'yToe', 'Dune Width', '2016 Dune Volume', 'xToe', 'Beach Slope', 'Beach Width', '2016 Beach Volume', 'xMHW']
    ll = ['D$_{high}$', 'D$_{low}$', 'Dune Width', 'Dune Volume', 'D$_{low,x}$', 'Beach Slope', 'Beach Width', 'Beach Volume', 'Shoreline']
    units = ['m NAVD88', 'm NAVD88', 'm', 'm$^{3}$/m', 'm', 'm/m', 'm', 'm$^{3}$/m', 'm']
    lims = [(None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None), (None, None)]
    err = [0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01, 0.01]

    # Make the DataFrame
    df = pd.DataFrame.from_dict({'Model Name': mm,
                                 'Field Name': fm,
                                 'Label': ll,
                                 'Unit': units,
                                 'Limits': lims,
                                 'Error': err})
    df.set_index('Model Name', inplace=True)

    # Calculate the uncertainty for each metric. Some parameters
    # share an uncertainty. The RTK errors are 0.01m (Y) and
    # 0.016m (X). Formulations based on file:///C:/Users/Geology/AppData/Local/Temp/10515.pdf
    just_vert = ['Y Crest', 'Y Toe']
    just_horizontal = ['X Toe', 'X MHW']
    vert_horiz = ['Dune Width', 'Dune Volume', 'Beach Slope', 'Beach Width', 'Beach Volume']
    dt = 2017 - 2016
    df['Error'].loc[just_vert] = np.sqrt(2 * (0.01**2)) / dt
    df['Error'].loc[just_horizontal] = np.sqrt(2 * (0.016 ** 2)) / dt
    df['Error'].loc[vert_horiz] = np.sqrt(2 * (np.sqrt((0.01**2) + (0.016**2)))**2) / dt

    return df


"""
Functions to make text files
"""


def error_metrics(df):
    """
    Print out a summary of the error metrics for each profile to
    a .csv file

    df: DataFrame with error metrics
    """

    # Get the profile numbers
    profiles = df['Profile'].unique()

    # Loop over the profiles
    for p in profiles:

        # Get row with the best (lowest RMSE) hindcast result
        temp_df = df.loc[df['Profile'] == p]
        best = temp_df.loc[temp_df['RMSE'] == np.nanmin(temp_df['RMSE'])].iloc[-1]

        # Save the DataFrame to a .csv
        fname = os.path.join(TEXT_DIR, f'BGB{p} Summary.csv')
        best.to_csv(fname)
        print(f'File Saved: {fname}')


"""
Functions to make figures
"""


def compare_orientations(save=False):
    """
    Plot the orientations of each profile on a polar plot
    to compare them

    save: Display the figure (False) or save and close it (True)
    """

    # Setup the figure
    fig = plt.figure(figsize=(inches, inches), dpi=dpi)
    ax = plt.subplot(111, polar=True)
    cmap = plt.cm.viridis(np.linspace(0, 1, len(range(1, 24))))
    profiles = range(1, 23)
    y, dx, dy = 10, 0, -9
    width = 0.0125

    # Add a grid
    ax.grid(color='lightgrey', linewidth=0.5, zorder=0)

    # Loop through the profiles
    loc_dict = dict()
    for pp in profiles:

        # Load the profile geo data
        fname = os.path.join(FIELD_DIR, '2016', f'BGB{pp}_cross.csv')
        df = pd.read_csv(fname, delimiter=',', header=None, names=['X', 'Y', 'Z', 'Zsmooth', 'D', 'Lon', 'Lat'])

        # Calculate the bearing of the profile
        lat0, lon0 = df['Lat'].iloc[0], df['Lon'].iloc[0]
        latf, lonf = df['Lat'].iloc[-1], df['Lon'].iloc[-1]
        loc_dict[f'BGB{pp}'] = [lat0, lon0, latf, lonf]
        geo_data = Geodesic.WGS84.Inverse(lat0, lon0, latf, lonf)
        bearing = geo_data['azi1']
        # bearing_adjusted = (bearing + 270) % 360
        bearing_adjusted = (450 - bearing + 180) % 360

        # Plot the bearing
        ax.arrow(np.deg2rad(bearing_adjusted), y, dx, dy, width=width, color=cmap[pp], zorder=pp)
        if pp == 15:
            ax.arrow(np.deg2rad(bearing_adjusted), y, dx, dy, width=width * 4, color='red', zorder=40, label='BGB15')
        elif pp == 22:
            ax.arrow(np.deg2rad(bearing_adjusted), y, dx, dy, width=width * 4, color='green', zorder=40, label='BGB22')

    # Calculate the distance between BGB15 and BGB22 using
    # the seaward most point of each. Set it as the title
    bgb_dist = Geodesic.WGS84.Inverse(loc_dict['BGB15'][2], loc_dict['BGB15'][3], loc_dict['BGB22'][2], loc_dict['BGB22'][3])
    title_str = f'BGB15 and BGB22 are {np.around(bgb_dist["s12"] / 1000, decimals=2)} km apart'
    ax.set_title(title_str, **font)

    # Set the X-Axis
    ax.set_xticklabels(['E', 'NE', 'N', 'NW', 'W', 'SW', 'S', 'SE'])
    ax.tick_params(axis='x', labelsize=font['fontsize'])

    # Set the Y-Axis
    ax.set_ylim(0, 10)
    ax.set_yticklabels([])

    # Save and close the figure
    title = f'BGB15 and BGB22 Orientations'
    save_and_close(fig, title, save)


def iribarren_boxplots(a, b, save=False):
    """
    Make a boxplot of Iribarren numbers measured for both profiles

    a: WindsurfResult object for BGB15
    b: WindsurfResult object for BGB22
    save: Display the figure (False) or save and close it (True)
    """

    # Setup the figure
    fig, ax = plt.subplots(nrows=1, figsize=(inches, inches), dpi=dpi)
    labels = ['Dissipative', 'Intermediate', 'Reflective']
    label_vals = [0, 0.3, 1.25]

    # Add lines for the different classifications (dissipative, reflective)
    ax.axhline(y=0.30, color='black', linestyle='--', linewidth=2, zorder=0)
    ax.axhline(y=1.25, color='black', linestyle='--', linewidth=2, zorder=0)

    # Plot the data
    ax.boxplot([a['Iribarren'], b['Iribarren']],
               showmeans=True,
               meanprops=dict(marker='d', markerfacecolor='blue', markeredgecolor='black', linewidth=0.25, markersize=4))

    # Find the mode for each profile and plot it
    # counter_bgb15, counter_bgb22 = Counter(a['Iribarren']), Counter(b['Iribarren'])
    # mode_bgb15, mode_bgb22 = counter_bgb15.most_common(1)[0][0], counter_bgb22.most_common(1)[0][0]
    # ax.scatter(x=[1, 2], y=[mode_bgb15, mode_bgb22], marker='s', color='red', edgecolor='black', linewidth=0.5, zorder=20)

    # Label the regions
    for lab, why in zip(labels, label_vals):
        ax.text(x=1.15, y=why + 0.05, s=lab, **font)

    # Set the X-Axis
    ax.set_xticklabels(['BGB15', 'BGB22'])

    # Set the Y-Axis
    ax.set_ylim(bottom=0, top=1.8)
    ax.set_ylabel('Iribarren Number', **font)

    # Save and close the figure
    title = f'BGB15 and BGB22 Iribarren Numbers'
    save_and_close(fig, title, save)


def model_v_field_profiles(pp, nn, dec=2, compare=False, save=False):
    """
    Plot the best hindcast profile overlain on the field profiles.
    Include the RMSE score and the BSS score

    pp: Windsurf result object
    nn: Int with the profile number
    dec: Int with the number of decimal places to round to (Default: 2)
    compare: Add a second subplot showing the dZ for the simulations and field profile (Default: False)
    save: Display the profile (False) or save and close it (True)
    """

    # Identify the best profile
    summary = pp.summary
    summary['Mean RMSE'] = summary[['Dune RMSE', 'Beach RMSE']].mean(axis=1)
    summary['Mean BSS'] = summary[['Dune BSS', 'Beach BSS']].mean(axis=1)
    best = summary.loc[summary['RMSE'] == np.nanmin(summary['RMSE'])].iloc[-1]
    best_profile = pp.profiles[f'{best["Generation"]}_{best["Simulation"]}']

    # Setup the figure
    if compare:
        fig, (ax1, ax2) = plt.subplots(nrows=2, figsize=(inches * 2, inches * 1.5), sharex='all', dpi=dpi,
                                       gridspec_kw={'height_ratios': [2, 1]})
        axes = [ax1, ax2]
    else:
        fig, ax1 = plt.subplots(nrows=1, figsize=(inches * 2, inches), dpi=dpi)
        axes = [ax1]

    # Add a grid
    [ax.grid(color='lightgrey', linewidth=0.5, zorder=0) for ax in axes]

    # Add the MHW line to the top plot and a zero-line to the bottom
    ax1.axhline(y=0.34, color='darkblue', linestyle='--', linewidth=2, zorder=2)
    ax1.text(x=1, y=0.35, s='MHW', color='darkblue', zorder=2, **font)
    if compare:
        ax2.axhline(y=0, color='black', linestyle='--', linewidth=2, zorder=2)

    # Add the fence
    if pp.adjusts['Fence'] is not None:
        [ax.axvline(x=pp.adjusts['Fence'], color='darkgray', linewidth=2, linestyle='--', zorder=2, label='Fence') for ax in axes]

    # Plot the field profile on top
    ax1.plot(pp.profiles['X'], pp.profiles['Field Init'], color='black', linewidth=2, linestyle='--', zorder=4, label='Field$_{2016}$')
    ax1.plot(pp.profiles['X'], pp.profiles['Field Final'], color='black', linewidth=2, linestyle='-', zorder=6, label='Field$_{2017}$')
    ax1.plot(pp.profiles['X'], best_profile, color='red', linewidth=2, linestyle='-', zorder=6, label='Model$_{f}$')

    # Add a legend and text labels on top
    ax1.legend(loc='upper right', fancybox=False, edgecolor='black', title=f'BGB{nn}')
    lab = f'RMSE = {np.around(best["RMSE"], decimals=dec)} m\nBSS = {np.around(best["BSS"], decimals=dec)} ({best["BSS Label"]})'
    ax1.text(x=0.1, y=pp.adjusts['Top'] * 0.82, s=lab, zorder=2, **font)

    # Plot the profile differences on the bottom
    if compare:
        ax2.plot(pp.profiles['X'], pp.profiles['Field Final'] - pp.profiles['Field Init'], linewidth=2, color='black', zorder=4, label='Field')
        ax2.plot(pp.profiles['X'], best_profile - pp.profiles['Model Init'], linewidth=2, color='red', label='Model')
        ax2.legend(loc='upper left', fancybox=False, edgecolor='black')

    # Set the X-Axis
    if compare:
        ax2.set_xlim(left=0, right=pp.adjusts['Right'])
        ax2.set_xlabel('Cross-Shore Distance (m)', **font)
    else:
        ax1.set_xlim(left=0, right=pp.adjusts['Right'])
        ax1.set_xlabel('Cross-Shore Distance (m)', **font)

    # Set the Y-Axis
    ax1.set_ylim(bottom=-1, top=pp.adjusts['Top'])
    ax1.set_ylabel('Elevation (m NAVD88)', **font)
    if compare:
        if nn == 15:
            ax2.set_ylim(bottom=-1, top=1)
        elif nn == 22:
            ax2.set_ylim(bottom=-0.5, top=1.5)
        ax2.set_ylabel('$\Delta$Z (m)', **font)

    # Save and close the figure
    if compare:
        title = f'BGB{nn} Model v Field Results (dZ)'
    else:
        title = f'BGB{nn} Model v Field Results'
    save_and_close(fig, title, save)


def morpho_time_series(res, metric, ylabel, field, f_err=None, ylim=(None, None), save=False):
    """
    Plot out a time series of some metric value from the
    Windsurf simulation and compare it to the change in the
    field data

    res: WindsurfResult object
    metric: String with the value to plot
    ylabel: String with the Y-Axis label
    field: String with the label for the field metric to plot
    f_err: Add a shaded region for the RTk uncertainty
    ylim: Tuple with Y-Axis limits
    save: Display the figure (False) or save and close (True)
    """

    # Pull out the field data
    field_data = res.field_morpho[field]
    field_data_0 = field_data.loc[res.field_morpho['Year'] == 2016].values[0]
    field_data_f = field_data.loc[res.field_morpho['Year'] == 2017].values[0]
    f0_hi, f0_lo = field_data_0 + f_err, field_data_0 - f_err
    ff_hi, ff_lo = field_data_f + f_err, field_data_f - f_err

    # Setup the figure
    fig, ax = plt.subplots(dpi=dpi, figsize=(inches * 2, inches))
    x = res.enviro['Times'][:len(res.morpho[metric])]
    x0, xf = x.values[0], x.values[-1]

    # Add a grid
    ax.grid(color='lightgrey', linewidth=0.5, zorder=0)

    # Plot the field data
    ax.fill_between([x0, xf], [f0_hi, ff_hi], [f0_lo, ff_lo], facecolor='red', edgecolor=None, alpha=0.15, zorder=2)
    ax.plot([x0, xf], [field_data_0, field_data_f], color='red', linewidth=2, zorder=4, label='Field')

    # Plot the model data
    ax.plot(x, res.morpho[metric], color='blue', linewidth=2, zorder=6, label='Model')

    # Add a legend
    ax.legend(loc='upper left', fancybox=False, edgecolor='black')

    # Set the X-Axis
    ax.set_xlabel('Date', **font)

    # Set the Y-Axis
    ax.set_ylim(ylim)
    ax.set_ylabel(ylabel, **font)

    # Save and close the figure
    title = f'BGB{res.p} {metric} Time Series'
    save_and_close(fig, title, save)


def nn_training_results(save=False):
    """
    Make a boxplot showing the best GA results for each sample size
    separated by the profile number

    save: Display the figure (False) or save and close it (True)
    """

    # Load the data
    fname = os.path.join(TEXT_DIR, 'NN calibration Tests Results.csv')
    df = pd.read_csv(fname, delimiter=',', header=0)
    df = df.loc[df['Samples'] <= 1300]

    # Setup the figure
    fig, (ax1, ax2) = plt.subplots(nrows=2, dpi=dpi, sharex='all', figsize=(inches * 2, inches * 2))

    # Make the plots
    g = sns.boxplot(x='Samples', y='GA Best', hue='Profile', data=df, ax=ax1)
    sns.boxplot(x='Samples', y='R2 Values', hue='Profile', data=df, ax=ax2)

    # Remove the legend from ax1
    g.legend_.remove()

    # Set the X-Axes
    ax1.set_xlabel('')
    ax2.set_xlabel('# Samples', **font)
    ax2.tick_params(axis='x', labelrotation=45)

    # Set the Y-Axes
    [ax.set_ylim(bottom, 1) for ax, bottom in zip([ax1, ax2], [0, -2])]
    [ax.set_ylabel(label, **font) for ax, label in zip([ax1, ax2], ['GA Best RMSE (m)', 'NN1 r$^{2}$'])]

    # Save and close the figure
    title = f'NN Training Tests Results'
    save_and_close(fig, title, save)


def param_bars(mdf, fdf, params, max_year=2017, save=False):
    """
    Make a barplot comparing changes in a parameter value between
    the field and model results

    mdf: DataFrame with morphometrics from the model
    fdf: DataFrame with morphometrics from the field
    params: dataFrame with parameter information
    save: Display the figure (False) or save and close it (True)
    """

    # Setup the figure
    fig, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9)) =\
        plt.subplots(nrows=3, ncols=3, dpi=dpi, figsize=(inches * 2, inches * 2))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8, ax9]

    # Loop through the parameters
    for ax, param in zip(axes, params.index):

        # Set values
        field_param = params['Field Name'].loc[param]
        lims = params['Limits'].loc[param]

        # Filter out the field data
        fdf = fdf.loc[fdf['Year'] <= max_year]

        # Get the unique years and profiles
        years = fdf['Year'].unique()
        profiles = mdf['Profile'].unique()

        # Make a DataFrame with the change for each profile
        # for the field and model
        df = change_df(mdf, fdf, param, field_param, years, profiles)

        # Plot the data
        g = sns.barplot(x='Profile', y='Results',
                        hue='Type', hue_order=['Model', 'Field'], palette=['cornflowerblue', 'lightgrey'],
                        edgecolor='black',
                        yerr=params['Error'].loc[param],
                        data=df, ax=ax)
        g.legend_.remove()

        # Set the X-Axis
        ax.set_xlabel('')

        # Set the Y-Axis
        ax.set_ylim(lims)
        ax.set_ylabel(f'$\Delta${params["Label"].loc[param]} ({params["Unit"].loc[param]})', **font)

    # Put a legend in the top left plot
    ax1.legend(loc='upper left', fancybox=False, edgecolor='black')

    # Label the X-Axes in thew bottom row
    [ax.set_xlabel('Profile', **font) for ax in [ax7, ax8, ax9]]

    # Save and close the figure
    title = f'Model Versus Field Results'
    save_and_close(fig, title, save)


def save_and_close(fig, title, save):
    """
    Add a transparent mask, then save
    and close the figure

    fig: Figure object to save
    title: String with the name to save the figure file as
    save: Save the figure (True) or close it (False)
    """

    if save:

        # Set a tight layout and a transparent background
        plt.tight_layout()
        fig.patch.set_color('w')
        fig.patch.set_alpha(0.0)

        # Save and close the figure
        title_w_extension = os.path.join(FIGURE_DIR, f'{title}.png')
        plt.savefig(title_w_extension, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi='figure')
        print('Figure saved: %s' % title_w_extension)
        plt.close()

    else:
        plt.show()


"""
Main analysis
"""

def main():
    """
    Run the analysis

    The result objects contain the following objects:
        - Adjusts: Dict with profile adjustments and profile boundaries
        - Enviro: DataFrame with envrionmental forcings used with the model
        - Field Morpho: DataFrame with morphometrics measured from the field profile for all years (2016-2020)
        - Morpho: Morphometrics from each time step of the simulation
        - Profiles: DataFrame with all of the final profiles for all of the simulations
        - Summary: DataFrame with the parameters and error scores for all simulations
    """

    # Load up the results
    new = False
    # bgb1 = ws.load_windsurf_result(p=1, t0=2016, tf=2017, dec=4, new=new)
    # bgb3 = ws.load_windsurf_result(p=3, t0=2016, tf=2017, dec=4, new=new)
    bgb15 = ws.load_windsurf_result(p=15, t0=2016, tf=2017, dec=4, new=new)
    bgb22 = ws.load_windsurf_result(p=22, t0=2016, tf=2017, dec=4, new=new)

    # Get a DataFrame with parameter information
    params_info = params_df()

    # Put all of the DataFrames that can be merged together into one
    dfs = [bgb15, bgb22]
    params = pd.concat([dd.summary for dd in dfs])
    f_morpho = pd.concat([dd.field_morpho for dd in dfs])
    m_morpho = pd.DataFrame()
    for pp, df in zip([15, 22], [bgb15.morpho, bgb22.morpho]):

        df['Profile'] = pp
        m_morpho = pd.concat([m_morpho, df])

    # Print out the error metrics for each hindcast
    error_metrics(df=params)

    # Plot the best hindcast results
    model_v_field_profiles(pp=bgb15, nn=15, compare=True, save=True)
    model_v_field_profiles(pp=bgb22, nn=22, compare=True, save=True)

    # Compare the changes in key metrics between field and model results
    param_bars(m_morpho, f_morpho, params_info, save=True)

    # Calculate the irribarren numbers for both profiles
    n_bgb15, n_bgb22 = irribarren(bgb15), irribarren(bgb22)
    iribarren_boxplots(n_bgb15, n_bgb22, save=True)

    # Plot the profiles on a polar plot to compare their orientations
    compare_orientations(save=True)

    # Plot the results from the NN training tests
    nn_training_results(save=True)


if __name__ == '__main__':
    main()
