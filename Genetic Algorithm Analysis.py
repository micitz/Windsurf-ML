"""
Code to analyze output from the genetic algorithm
calibration tests

Michael Itzkin, 2/16/2021
"""

from Windsurf_GA_Result import load_windsurf_result

from sklearn.linear_model import LinearRegression
from scipy import optimize

from itertools import combinations
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os

# General settings
font = {'fontname': 'Arial', 'fontweight': 'normal', 'fontsize': 14}
dpi = 300
inches = 3.8
TEXT_DIR = os.path.join('..', 'Text Files')
FIELD_DIR = os.path.join('..', '..', 'Bogue Banks Field Data')
FIGURE_DIR = os.path.join('..', 'Figures', 'Genetic Algorithm Figures')



"""
Functions to make figures
"""


def bss_number(row):
    if row == 'Very Poor':
        return 0
    elif row == 'Poor':
        return 1
    elif row == 'Fair':
        return 2
    elif row == 'Good':
        return 3
    elif row == 'Excellent':
        return 4


def save_and_close(fig, title, save, tight=True):
    """
    Give the figure a tight and transparent background
    then close it

    fig: Figure object
    title: String with the title of the figure
    save: Save the figure (True) of display it (False)
    """

    # Set a tight layout and a transparent background
    if tight:
        plt.tight_layout()
    fig.patch.set_color('w')
    fig.patch.set_alpha(0.0)

    # Save and close the figure
    if save:
        title_w_extension = os.path.join(FIGURE_DIR, f'{title}.png')
        plt.savefig(title_w_extension, bbox_inches='tight', facecolor=fig.get_facecolor(), dpi='figure')
        print('Figure saved: %s' % title_w_extension)
        plt.close()
    else:
        plt.show()


def y_v_x(df, p, color='RMSE', save=False):
    """
    Make a single figure showing all parameters as a function of one parameter
    colored by their qualitative score

    df: DataFrame with the parameter values and labels
    p: List of parameters (including y)
    color: String with the column to color the points by
    profile: Int with the profile number to analyze (Default = None)
    save: Bool to display (False) or save (True) the figure (Default = False)
    """

    # Setup the figure
    fig, axes = plt.subplots(nrows=8, ncols=8, sharex='col', figsize=(inches * 2, inches * 2), dpi=dpi,
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    axes = axes.ravel().tolist()
    remove_axes = [1, 2, 3, 4, 5, 6, 7, 10, 11, 12, 13, 14, 15, 19, 20,
                   21, 22, 23, 28, 29, 30, 31, 37, 38, 39, 46, 47, 55]
    left_axes = [0, 8, 16, 24, 32, 40, 48, 56]
    bottom_axes = [56, 57, 58, 59, 60, 61, 62, 63]
    kde_axes = [0, 9, 18, 27, 36, 45, 54, 63]
    if 'RMSE' in color:
        cmap = 'viridis_r'
        vmin, vmax = 0, 1
    elif 'BSS' in color:
        df[color] = df[f'{color} Label'].apply(bss_number)
        cmap = plt.cm.get_cmap('plasma', 5)
        vmin, vmax = 0, 4

    # Hide un-necessary axes
    [axes[ii].axis('off') for ii in remove_axes]

    # Loop through the KDE Axes
    for ii, param in zip(kde_axes, p):
        sns.kdeplot(df[param], ax=axes[ii], color='blue', shade=True, legend=False)

    # Plot the Cb v. X row
    plot = axes[8].scatter(x=df['m'], y=df['Cb'], s=10, c=df[color], cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

    # Plot the facAs v. X row
    for ax, col in zip([16, 17], ['m', 'Cb']):
        axes[ax].scatter(x=df[col], y=df['facAs'], s=10, c=df[color], cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

    # Plot the facSk v. X row
    for ax, col in zip([24, 25, 26], ['m', 'Cb', 'facAs']):
        axes[ax].scatter(x=df[col], y=df['facSk'], s=10, c=df[color], cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

    # Plot the lsgrad v. X row
    for ax, col in zip([32, 33, 34, 35], ['m', 'Cb', 'facAs', 'facSk']):
        axes[ax].scatter(x=df[col], y=df['lsgrad'], s=10, c=df[color], cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

    # Plot the bedfricoef v. X row
    for ax, col in zip([40, 41, 42, 43, 44], ['m', 'Cb', 'facAs', 'facSk', 'lsgrad']):
        axes[ax].scatter(x=df[col], y=df['bedfric'], s=10, c=df[color], cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

    # Plot the wetslp v. X row
    for ax, col in zip([48, 49, 50, 51, 52, 53], ['m', 'Cb', 'facAs', 'facSk', 'lsgrad', 'bedfric']):
        axes[ax].scatter(x=df[col], y=df['wetslp'], s=10, c=df[color], cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

    # Plot the veg density v. X row
    for ax, col in zip([56, 57, 58, 59, 60, 61, 62], ['m', 'Cb', 'facAs', 'facSk', 'lsgrad', 'bedfric', 'wetslp']):
        axes[ax].scatter(x=df[col], y=df['vegDensity'], s=10, c=df[color], cmap=cmap, vmin=vmin, vmax=vmax, zorder=2)

    # Add a colorbar
    # fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.72, 0.32, 0.05, 0.55])
    cbar = plt.colorbar(plot, ticks=[0, 0.5, 1.0, 1.5, 2.0], cax=cbar_ax)
    cbar.ax.tick_params(labelsize=8)
    if 'RMSE' in color:
        cbar.set_label(f'{color} (m)', **font)
        cbar.set_ticks([0, 0.25, 0.50, 0.75, 1])
    elif 'BSS' in color:
        cbar.set_label(f'{color}', **font)
        cbar.set_ticks(np.linspace(0.4, 3.6, 5))
        cbar.set_ticklabels(['Very Poor (<0)', 'Poor (0 - 0.3)', 'Fair (0.3 - 0.6)', 'Good (0.6 - 0.8)', 'Excellent (>0.8)'])

    # Set the X-Axes
    [axes[ii].set_xlabel(col, fontsize=8) for ii, col in zip(bottom_axes, p)]

    # Set the Y-Axes
    [axes[ii].set_ylabel(col, fontsize=8) for ii, col in zip(left_axes, p)]

    # Clear the ticks and labels for non "edge" subplots
    for ii in range(0, len(axes)):
        if ii not in left_axes and ii not in bottom_axes:
            axes[ii].set_yticks([])
    [axes[ii].set_yticks([]) for ii in bottom_axes[1:]]

    # Save and close the figure
    title = os.path.join('Param Figures', f'Parameter Phase Diagrams (by {color})')
    save_and_close(fig=fig, title=title, save=save, tight=False)


"""
Functions to make tables and text files
"""


def all_params(results, profiles, longs):
    """
    Combine the summary dataframes from all of the Windsurf
    results objects into a single DataFrame. Return the DataFrame
    and save a .csv of it
    """

    # Make a generic DataFrame to loop into
    df = pd.DataFrame()

    # Loop through the individual Windsurf results and
    # concatenate into the larger DataFrame
    for res, pp, long in zip(results, profiles, longs):
        temp = res.summary
        temp['Profile'], temp['Long'] = pp, long
        df = pd.concat([df, temp])

    # Save a .csv file
    fname = os.path.join(TEXT_DIR, 'All Genetic Algorithm Parameters.csv')
    df.to_csv(fname, sep=',', index=False)
    print(f'File Saved: {fname}')

    return df


def best_params(res, profiles, longs, params):
    """
    Make a table with the parameters for the
    best run from each experiment

    res: DataFrame with all the parameters from the runs
    profiles: List with ints of the profile numbers
    longs: List of Bools to use the long or short sims
    params: List of strings with the parameters
    """

    # Set an empty DataFrame
    df = pd.DataFrame()

    # Loop through the results
    for pp, ll in zip(profiles, longs):

        # Pull out the relevant part of the results DataFrame
        rel = res.loc[(res['Profile'] == pp) & (res['Long'] == ll)]
        rel.drop(['Folder', 'Profile', 'Long'], axis=1, inplace=True)
        rel = rel.reset_index(drop=True)

        # Add a column for the name
        if ll:
            rel['Hindcast'] = f'BGB{pp} (2016-2019)'
        else:
            rel['Hindcast'] = f'BGB{pp} (2016-2017)'

        # Put the row into the main DataFrame
        row = rel[rel['RMSE'] == rel['RMSE'].min()].iloc[0]
        df = pd.concat([df, row], axis=1)

    # Order the columns
    columns = ['Hindcast']
    for param in params:
        columns.append(param)
    for col in ['RMSE', 'Beach RMSE', 'Dune RMSE', 'Generation', 'Simulation']:
        columns.append(col)
    df = df.T
    df = df[columns]

    # Save a .csv
    filename = os.path.join(TEXT_DIR, 'Genetic Algorithm Best Parameterizations.csv')
    df.to_csv(filename, sep=',', index=False)
    print(f'File Saved: {filename}')


def combined_params(bgb15, bgb22):
    """
    Put the results from the combined BGB15 and BGB22 calibration
    runs into a single DataFrame. Pull the parameter values and run
    numbers from one of them and split the rest. Label the individual
    RMSE values and make additional "mean" RMSE values.

    bgb15: WindsurfGAResult object for BGB15
    bgb22: WindsurfGAResult object for BGB22
    """

    # Make a new DataFrame
    df = pd.DataFrame()

    # Set the columns that are unique and that will be different
    same = ['Generation', 'Simulation', 'm', 'Cb', 'facAs', 'facSk', 'lsgrad', 'bedfric', 'wetslp', 'vegDensity']
    diff = ['RMSE', 'Dune RMSE', 'Beach RMSE', 'Folder']

    # Take the "same" columns from BGB15 (or BGB22, doesn't matter) and
    # put them in the new DataFrame
    df[same] = bgb15.summary[same]

    # Take the "diff" columns and add a "BGB15" and "BGB22"
    # version in the new DataFrame
    for p, temp in zip([15, 22], [bgb15.summary, bgb22.summary]):
        for col in diff:
            df[f'BGB{p} {col}'] = temp[col]

    # Make "mean" version of the RMSE columns
    for col in diff[:-1]:
        df[f'Mean {col}'] = df[[f'BGB15 {col}', f'BGB22 {col}']].mean(axis=1)

    # Save a .csv file
    fname = os.path.join(TEXT_DIR, 'All Combined Genetic Algorithm Parameters.csv')
    df.to_csv(fname, sep=',', index=False)
    print(f'File Saved: {fname}')

    return df


def func(x, coeffs):
    """
    RMSE = f(params)

    x: List of values to calibrate
    coeffs: List of coefficients to use in the function
    """

    return abs((coeffs[0] * x[0]) + (coeffs[1] * x[1]) + (coeffs[2] * x[2]) + (coeffs[3] * x[3]) +\
           (coeffs[4] * x[4]) + (coeffs[5] * x[5]) + (coeffs[6] * x[6]) + (coeffs[7] * x[7]) + coeffs[8])


def multi_lin_reg(df, bounds, dec=2):
    """
    Perform a multiple linear regression on the parameters
    to identify a function that relates the RMSE score to
    the calibrated parameters. Print results to a text file

    df: DataFrame with all parameters
    bounds: Dict with parameters and their boundary values
    dec: Int with the number of decimals to round to (Default = 2)
    """

    # Open the text file
    fname = os.path.join(TEXT_DIR, 'Multiple Linear Regression Results.txt')
    f = open(fname, 'w+')

    # Loop through the long and short parameterizations
    for long in [False]:

        # Write a section header
        if long:
            f.write('### 2016-2019 ###:\n')
        else:
            f.write('### 2016-2017 ###:\n')

        # Perform the regression
        regr = LinearRegression(fit_intercept=True)
        use = df.loc[df['Long'] == long].dropna()
        regr.fit(use[list(bounds.keys())], use['RMSE'])
        r_squared = regr.score(use[list(bounds.keys())], use['RMSE'])

        # Perform the optimization
        opt_coeffs = list(regr.coef_)
        opt_coeffs.append(regr.intercept_)
        x0 = [np.mean(x) for x in bounds.values()]
        opt = optimize.minimize(func, x0=x0, args=opt_coeffs, bounds=list(bounds.values()))

        # Write the equation
        f.write('RMSE = {}m + {}Cb + {}facAs + {}facSk + {}lsgrad + {}bedfric + {}wetslp + {}vegDensity + {}\n'.format(
            np.around(regr.coef_[0], decimals=dec),
            np.around(regr.coef_[1], decimals=dec),
            np.around(regr.coef_[2], decimals=dec),
            np.around(regr.coef_[3], decimals=dec),
            np.around(regr.coef_[4], decimals=dec),
            np.around(regr.coef_[5], decimals=dec),
            np.around(regr.coef_[6], decimals=dec),
            np.around(regr.coef_[7], decimals=dec),
            np.around(regr.intercept_, decimals=dec),
        ))

        # Write the R2 value
        f.write(f'R2 = {np.around(r_squared, decimals=4)}\n')

        # Write out the optimized RMSE value
        f.write(f'Minimized RMSE: {opt.fun}\n')

        # Write the parameter values for the minimized solution
        for param, value in zip(bounds.keys(), opt.x):
            f.write(f'{param}:\t{np.around(value, decimals=dec)}\n')

    # Close the file
    f.close()
    print(f'File Saved: {fname}')


"""
Run the analysis
"""


def main():
    """
    Run the analysis
    """

    """
    Set setting and prep for the analysis
    """

    # Turn sections on and off
    individual = True
    group = True

    # Load the results
    new = False
    bgb1 = load_windsurf_result(p=1, t0=2016, tf=2017, new=new)
    bgb3 = load_windsurf_result(p=3, t0=2016, tf=2017, new=new)
    bgb15 = load_windsurf_result(p=15, t0=2016, tf=2017, new=new)
    bgb22 = load_windsurf_result(p=22, t0=2016, tf=2017, new=new)

    # Set the parameters tested
    bounds = {'m': [0.00, 5.00],
              'Cb': [0.00, 3.00],
              'facAs': [0.00, 1.00],
              'facSk': [0.00, 1.00],
              'lsgrad': [-0.1, 0.1],
              'bedfric': [0.00, 0.10],
              'wetslp': [0.00, 1.00],
              'vegDensity': [0.00, 10.00]}

    # Combine the summary DataFrames
    results = [bgb1, bgb3, bgb15, bgb22]
    profiles = [1, 3, 15, 22]
    longs = [False, False, False, False]
    df = all_params(results, profiles, longs)

    """
    Analyze the individual algorithm results
    """
    if individual:
        for res in results:

            # Print out a summary of the simulations
            res.print_summary()

            # Make a boxplot of the RMSE values over each generation and
            # plot the RMSE calculation figures for every single simulation
            res.boxplot(save=True)
            # res.rmse_plot(save=True)

            # Make a boxplot of the parameters over each generation
            res.params_plot(save=True)

            # Plot an overlay of the final profiles for each generation
            res.generation_overlays(save=True)

            # Plot the average profile for all generations
            res.average_profiles(factor=1, save=True)

            # Plot the best result for all generations
            res.best_profiles(save=True)

    """
    Analyze all of the algorithm results together
    """
    if group:

        # Perform a multiple linear regression on the parameters
        # and optimize to find the ideal parameterization
        multi_lin_reg(df, bounds)

        # Make a table with the best parameters for the simulations
        best_params(df, profiles, longs, list(bounds.keys()))

        # Analyze the RMSE as a function of two parameters
        y_v_x(df.loc[df['Long'] == False], bounds.keys(), color='RMSE', save=True)
        y_v_x(df.loc[df['Long'] == False], bounds.keys(), color='BSS', save=True)
        y_v_x(df.loc[df['Long'] == False], bounds.keys(), color='Dune RMSE', save=True)
        y_v_x(df.loc[df['Long'] == False], bounds.keys(), color='Beach RMSE', save=True)
        y_v_x(df.loc[df['Long'] == False], bounds.keys(), color='Dune BSS', save=True)
        y_v_x(df.loc[df['Long'] == False], bounds.keys(), color='Beach BSS', save=True)


if __name__ == '__main__':
    main()
