"""
Make a windsurf genetic algorithm result object to
store all of the data from the simulations in an
organized way to analyze more easily

Michael Itzkin, 2/21/2021
"""

from scipy import io, stats
from collections import Counter
from scipy.interpolate import interp1d
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib.pyplot as plt
import seaborn as sns
import netCDF4 as nc
import pandas as pd
import numpy as np
import pickle
import os

# General settings
font = {'fontname': 'Arial', 'fontweight': 'normal', 'fontsize': 14}
dpi = 300
inches = 3.8
TEXT_DIR = os.path.join('..', 'Text Files')
FIELD_DIR = os.path.join('..', '..', 'Bogue Banks Field Data')
FIGURE_DIR = os.path.join('..', 'Figures', 'Genetic Algorithm Figures')


"""
Functions that operate outside the class but use it
"""


def load_windsurf_result(p, t0, tf, dec=4, special=None, new=True):
    """
    Load a windsurf result object either by creating a new one or
    from a pickle of an old one. The class object takes a long time
    to initialize so it is much faster to save and load a pickle of it so
    long as you don't want to store any new data.

    p: Int with the profile number being worked on
    t0: Int with the start year of the simulation
    tf: Int with the final year of the simulation
    dec: Int with the number of decimal places to round to
    special: String with the folder name of a combo calibration run
    new: Create a new object (True) or load a pickle of an existing one (False)
    """

    # Set the pickle fname
    pickle_fname = os.path.join('..', f'BGB{p} Genetic Algorithm', f'BGB{p} GA Result.pkl')

    if new:
        ws_result = WindsurfGAResult(p=p, t0=t0, tf=tf, dec=dec, special=special)
        with open(pickle_fname, 'wb') as output:
            pickle.dump(ws_result, output, pickle.HIGHEST_PROTOCOL)
    else:
        with open(pickle_fname, 'rb') as input:
            ws_result = pickle.load(input)

    return ws_result


"""
Functions called by the class
"""


def adjusts(p, tf=2017):
    """
    The cross-shore positions of profiles needs to
    be adjusted. This function just returns the right
    values to adjust by
    """

    if p == 1 and (tf == 2017):
        field_tf_adjust = 21
        tail = 50
        fence = None
        right = 120
        top = 7
    elif p == 1 and (tf == 2019):
        field_tf_adjust = 0
        tail = 50
        fence = None
        right = 120
        top = 7
    elif p == 3 and (tf == 2017):
        field_tf_adjust = -2
        tail = 0
        fence = 1
        right = 60
        top = 5
    elif p == 3 and (tf == 2019):
        field_tf_adjust = -11
        tail = 0
        fence = 1
        right = 60
        top = 5
    elif p == 15 and (tf == 2017):
        field_tf_adjust = 6
        tail = 0
        fence = 15
        right = 60
        top = 6
    elif p == 15 and (tf == 2019):
        field_tf_adjust = -3
        tail = 10
        fence = 15
        right = 40
        top = 6
    elif p == 19 and (tf == 2017):
        field_tf_adjust = 0
        tail = 0
        fence = 20
        right = 120
        top = 7
    elif p == 19 and (tf == 2017):
        field_tf_adjust = 0
        tail = 0
        fence = 20
        right = 120
        top = 7
    elif (p == 22) and (tf == 2017):
        field_tf_adjust = 2
        tail = 0
        fence = None
        right = 140
        top = 7
    elif (p == 22) and (tf == 2019):
        field_tf_adjust = -18
        tail = 0
        fence = None
        right = 140
        top = 6

    return {
        'dField': field_tf_adjust,
        'Tail': tail,
        'Fence': fence,
        'Right': right,
        'Top': top,
    }


def bss(df, dec=3, mask=None, column='Model Final'):
    """
    Calculate the BSS for the model run

    df: DataFrame with the gridded profiles in it
    dec: Number of decimals to round to (Default = 3)
    mask: Int with the values to use to mask out the profile
    """

    # Mask out values
    if mask is None:
        pass
    else:
        df = df.loc[df['Mask'] == mask]

    # Get the profiles
    z0 = df['Field Final']
    zm = df[column]
    zb = df['Field Init']

    # Calculate the BSS
    numerator = np.nansum(np.abs(z0 - zm)**2)
    denominator = np.nansum(np.abs(z0 - zb)**2)
    bss = 1 - (numerator / denominator)

    # Apply a label
    if bss < 0:
        label = 'Very Poor'
    elif 0 <= bss <= 0.3:
        label = 'Poor'
    elif 0.3 < bss <= 0.6:
        label = 'Fair'
    elif 0.6 < bss <= 0.8:
        label = 'Good'
    elif 0.8 < bss <= 1:
        label = 'Excellent'

    return np.around(bss, decimals=dec), label


def find_toe(df):
    """
    Find the toe position on the starting profile
    and adjust the profile mask so that the tail is
    all zeroes, the dune is all 1s and the beach is 2s

    df: DataFrame with the gridded profiles
    """

    # Find the dune crest index
    dhigh_ix = np.argwhere(df['Model Init'] == np.nanmax(df['Model Init']))[0][0]

    # Make a copy of the profile and make a straight line between
    # the dune crest and MHW
    p_copy = df['Model Init'].copy(deep=True)
    p_copy[:dhigh_ix] = np.linspace(start=df['Model Init'].iloc[0],
                                    stop=df['Model Init'].iloc[dhigh_ix],
                                    num=len(p_copy[:dhigh_ix]))

    # Subtract the original from the copy
    diff = p_copy - df['Model Init'].values

    # Take the index of the maximum value of "diff"
    # to set as the dune toe location
    toe_ix = np.argwhere(diff == np.nanmax(diff))[0][0]

    # Set the mask to be 2s up to the toe index
    df['Mask'].iloc[:toe_ix] = 2

    return df


def load_and_grid_profiles(DIR, p, t0, tf):
    """
    Load the field profile and model profiles and interpolate
    all onto the Windsurf X-Grid in order to analyze error metrics.

    Returns a DataFrame with the X-Grid, final and initial gridded
    profiles, and an array to mask out the "smooth" part of the profile

    DIR: Path to the windsurf.nc file
    p: Int with the profile number
    t0: Int with the year the hindcast starts at
    tf: Int with the year the hindcast ends at
    """

    # Adjust the profile to align them properly
    adjust_df = adjusts(p, tf=tf)

    # Load the model and pull out the X-Grid and
    # initial profile. This is the same for every
    # time simulation so just pull from the first
    # for simplicity
    filename = os.path.join(DIR, 'windsurf.nc')
    data = nc.Dataset(filename)
    x_grid = -data['x'][:]
    initial_profile = data['zb'][:, 0]
    final_profile = data['zb'][:, -1]
    data.close()

    # Load the first and last field profile
    columns = ['X', 'Y', 'Z', 'Zsmooth', 'D', 'Lat', 'Lon']
    init_field_fname = os.path.join(FIELD_DIR, f'{t0}', f'BGB{p}_cross.csv')
    final_field_fname = os.path.join(FIELD_DIR, f'{tf}', f'BGB{p}_cross.csv')
    init_field = pd.read_csv(init_field_fname, header=None, delimiter=',', names=columns).dropna()
    final_field = pd.read_csv(final_field_fname, header=None, delimiter=',', names=columns).dropna()

    # Push the field profiles back to 0
    init_field['D'] -= init_field['D'].iloc[0] + adjust_df['Tail']
    final_field['D'] -= final_field['D'].iloc[0] - adjust_df['dField'] + adjust_df['Tail']

    # Cut the windsurf profile off at the seaward-most point
    # of the final field profile
    field_max = np.nanmax(final_field['D'])
    final_profile = final_profile[(x_grid <= field_max) & (x_grid >= 0)]
    initial_profile = initial_profile[(x_grid <= field_max) & (x_grid >= 0)]
    x_grid = x_grid[(x_grid <= field_max) & (x_grid >= 0)]

    # Interpolate the field profiles onto the Windsurf grid
    init_interp = interp1d(init_field['D'], init_field['Zsmooth'], fill_value='extrapolate')
    init_field_grid = init_interp(np.array(x_grid))
    final_interp = interp1d(final_field['D'], final_field['Zsmooth'], fill_value='extrapolate')
    final_field_grid = final_interp(np.array(x_grid))

    # The landward boundary on the Windsurf profile is smoothed
    # out which is not a real feature. This will mess up the error
    # scores if it is included, so ID where it starts so it can be
    # easily ignored when analyzing the run
    init_model_z = initial_profile[-1]
    eps = 0.05
    for ii in range(len(x_grid) - 1, 0, -1):
        if initial_profile[ii] > init_model_z + eps:
            break
        else:
            start_ix = ii

    # Make an array of 0s and 1s with 0s being all points
    # landward of start_ix and 1s being all points
    # seaward of start_ix
    smooth_mask = np.zeros(shape=(len(x_grid), 1))[:]
    smooth_mask[:start_ix] = 1

    # Place the gridded profiles into a DataFrame and
    # add a mask for the beach
    df = pd.DataFrame.from_dict({
        'X': x_grid,
        'Model Init': initial_profile,
        'Model Final': final_profile,
        'Field Init': init_field_grid,
        'Field Final': final_field_grid,
        'Mask': smooth_mask[:, 0],
    })
    find_toe(df)

    return df


def make_profiles_df(summ_df, p, t0, tf, special=None):
    """
    Make a DataFrame with the gridded and masked model
    and field profile

    summ_df: DataFrame with summaries of the runs
    p: Int with the profile number
    t0: Int with the year the hindcast starts at
    tf: Int with the year the hindcast ends at
    special: String with the folder
    """

    # Set an empty DataFrame to loop into
    profiles_df = pd.DataFrame()

    for ix, row in summ_df.iterrows():
        if special is not None:
            USE_DIR = os.path.join('..', f'{special}')
        elif tf == 2017:
            USE_DIR = os.path.join('..', f'BGB{p} Genetic Algorithm')
        elif tf == 2019:
            USE_DIR = os.path.join('..', f'BGB{p} Genetic Algorithm Long')
        pp = load_and_grid_profiles(DIR=os.path.join(USE_DIR, row['Folder']), p=p, t0=t0, tf=tf)

        # The final model result is the only thing that differs between
        # simulations so eveything else can be pulled just from gen 1 run 1
        if ix == 0:
            profiles_df['X'] = pp['X']
            profiles_df['Field Init'] = pp['Field Init']
            profiles_df['Field Final'] = pp['Field Final']
            profiles_df['Model Init'] = pp['Model Init']
            profiles_df['Mask'] = pp['Mask']
        profiles_df[f'{row["Generation"]}_{row["Simulation"]}'] = pp['Model Final']

    return profiles_df


def make_summary_df(p, t0, tf, dec=4, special=None):
    """
    Compile results from the GA simulations and store
    them in a DataFrame for analysis

    p: Int with the profile to analyze
    t0: Int with the initial year of analysis
    tf: Int with the final year of analysis
    dec: Int with the number of decimals to round to (Default = 4)
    """

    # Set empty lists to loop into. These will be used to
    # construct the DataFrame at the end
    gens, runs, m, cb, facas_lo, facas_hi, facsk_lo, facsk_hi, lsgrad, bedfric, wetslp, beta,\
    veg_density, rmses, drmses, brmses, folders, bsses, dbsses, bbsses, bss_labels, dbss_labels, bbss_labels =\
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    # Set the path to the folder to work in
    if special is not None:
        USE_DIR = os.path.join('..', f'{special}')
    elif tf == 2017:
        USE_DIR = os.path.join('..', f'BGB{p} Genetic Algorithm')
    elif tf == 2019:
        USE_DIR = os.path.join('..', f'BGB{p} Genetic Algorithm Long')

    # The folders in the directory will have long names which are necessary for
    # the GA to run in parallel but a pain to work with as is. Calculate the
    # number of generations and simulations in each generation from the folder
    # titles. Save the full firectory list to later iterate through the folder names
    full_dirlist = [item for item in os.listdir(USE_DIR) if os.path.isdir(os.path.join(USE_DIR, item))]
    dirlist = [item.split('_')[0][3:] for item in full_dirlist]
    if '' in dirlist:
        dirlist.remove('')
    ngen, nsims = Counter(dirlist).keys(), Counter(dirlist).values()

    # Loop through the generations and folders
    # counter to count the number of runs
    for gen, gen_key in enumerate(ngen, 1):
        run = 0
        for folder in full_dirlist:
            if gen_key in folder:
                run += 1

                # Pull the parameter values that were
                # calibrated by the genetic algorithm
                if special is not None:
                    filename = os.path.join(USE_DIR, folder,  f'{p}', 'windsurf_setup.mat')
                else:
                    filename = os.path.join(USE_DIR, folder, 'windsurf_setup.mat')
                file = io.loadmat(filename, squeeze_me=True, struct_as_record=False)
                m.append(np.around(file['veg'].CDM.m, decimals=dec))
                beta.append(np.around(file['veg'].CDM.beta, decimals=dec))
                cb.append(np.around(file['sed'].Aeolis.Cb, decimals=dec))
                facas_lo.append(np.around(file['sed'].XB.facAsLo, decimals=dec))
                facas_hi.append(np.around(file['sed'].XB.facAsHi, decimals=dec))
                facsk_lo.append(np.around(file['sed'].XB.facSkLo, decimals=dec))
                facsk_hi.append(np.around(file['sed'].XB.facSkHi, decimals=dec))
                lsgrad.append(np.around(file['sed'].XB.lsgrad, decimals=dec))
                bedfric.append(np.around(file['flow'].XB.bedfriccoef, decimals=dec))
                wetslp.append(np.around(file['sed'].XB.wetslope, decimals=dec))
                veg_density.append(np.around(file['veg'].CDM.startingDensity, decimals=dec))

                # Calculate the RMSE values and Brier Skill Scores
                if special is None:
                    temp_dir = os.path.join(USE_DIR, folder)
                else:
                    temp_dir = os.path.join(USE_DIR, folder, f'{p}')
                df = load_and_grid_profiles(temp_dir, p, t0, tf)
                rmses.append(rmse(df.loc[df['Mask'] >= 1], dec=dec))
                drmses.append(rmse(df.loc[df['Mask'] == 1], dec=dec))
                brmses.append(rmse(df.loc[df['Mask'] == 2], dec=dec))

                if rmse(df.loc[df['Mask'] >= 1], dec=dec) == np.nan:
                    bsses.append(np.nan), bss_labels.append('Very Poor')
                else:
                    bscore, bss_label = bss(df.loc[df['Mask'] >= 1], dec=dec)
                    bsses.append(bscore), bss_labels.append(bss_label)

                if rmse(df.loc[df['Mask'] == 1], dec=dec) == np.nan:
                    dbsses.append(np.nan), dbss_labels.append('Very Poor')
                else:
                    dbss, dbss_label = bss(df.loc[df['Mask'] == 1], dec=dec)
                    dbsses.append(dbss), dbss_labels.append(dbss_label)

                if rmse(df.loc[df['Mask'] == 2], dec=dec) == np.nan:
                    bbsses.append(np.nan), bbss_labels.append('Very Poor')
                else:
                    bbss, bbss_label = bss(df.loc[df['Mask'] == 2], dec=dec)
                    bbsses.append(bbss), bbss_labels.append(bbss_label)

                # Update the lists
                gens.append(gen)
                runs.append(run)
                if special is not None:
                    folders.append(os.path.join(folder, f'{p}'))
                else:
                    folders.append(folder)

    return pd.DataFrame.from_dict({
        'Generation': gens,
        'Simulation': runs,
        'm': m,
        'Cb': cb,
        'facAs': facas_lo,
        'facSk': facsk_lo,
        'lsgrad': lsgrad,
        'bedfric': bedfric,
        'wetslp': wetslp,
        'vegDensity': veg_density,
        'RMSE': rmses,
        'Dune RMSE': drmses,
        'Beach RMSE': brmses,
        'BSS': bsses,
        'Dune BSS': dbsses,
        'Beach BSS': bbsses,
        'BSS Label': bss_labels,
        'Dune BSS Label': dbss_labels,
        'Beach BSS Label': bbss_labels,
        'Folder': folders,
    })


def rmse(df, dec=3, mask=None, column='Model Final'):
    """
    Calculate the RMSE for the model run

    df: DataFrame with the gridded profiles in it
    dec: Number of decimals to round to (Default = 3)
    mask: Int with the values to use to mask out the profile
    """

    # Mask out values
    if mask is None:
        pass
    else:
        df = df.loc[df['Mask'] == mask]

    # Calculate the RMSE
    y = df[column]
    yhat = df['Field Final']
    err = y - yhat
    square_error = err**2
    mse = np.nanmean(square_error)
    root_mse = np.sqrt(mse)
    return np.around(root_mse, decimals=dec)


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


"""
Make the WindsurfGAResult class
"""


class WindsurfGAResult():

    def __init__(self, p, t0, tf, dec=4, special=None):
        """
        Initialize the results object

        p: Int with the profile number being worked on
        t0: Int with the start year of the simulation
        tf: Int with the final year of the simulation
        dec: Int with the number of decimal places to round to
        special: String with the folder name of a combo calibration run
        """

        # Initialize input arguments
        self.p = p
        self.t0 = t0
        self.tf = tf
        self.dec = dec
        self.adjusts = adjusts(self.p, self.tf)
        if special is None:
            self.special = None
        else:
            self.special = f'{special} Genetic Algorithm'

        # Make a summary DataFrame with run parameters and error metrics
        self.summary = make_summary_df(self.p, self.t0, self.tf, self.dec, self.special)
        self.ngen = self.summary['Generation'].unique()

        # Make a profiles Dataframe with the gridded model and field profiles
        self.profiles = make_profiles_df(self.summary, self.p, self.t0, self.tf, self.special)


    """
    Class methods to analyze the data
    """

    def print_summary(self):
        """
        Print out a summary of the simulations to a text file
        """

        # Set the filename and open the file
        if self.special is None:
            filename = os.path.join(TEXT_DIR, f'BGB{self.p} {self.t0}-{self.tf} Genetic Algorithm Summary.txt')
        else:
            filename = os.path.join(TEXT_DIR, f'BGB{self.p} ({self.special}) Genetic Algorithm Summary.txt')
        f = open(filename, 'w+')

        # Print a header
        f.write(f'BGB{self.p} {self.t0}-{self.tf} Genetic Algorithm Summary:\n\n')

        # Loop through the generations and analyze
        for gen in self.ngen:
            use_df = self.summary.loc[self.summary['Generation'] == gen]
            nsims = len(use_df.index)
            if use_df['RMSE'].isnull().all():
                str = f'Generation {gen}: All NAN'
            else:
                mean_rmse = np.around(np.nanmean(use_df['RMSE']), decimals=self.dec)
                std_rmse = np.around(np.nanstd(use_df['RMSE']), decimals=self.dec)
                med_rmse = np.around(np.nanmedian(use_df['RMSE']), decimals=self.dec)
                best_rmse = np.around(np.nanmin(use_df['RMSE']), decimals=self.dec)
                best_run = use_df['Simulation'].loc[use_df['RMSE'] == np.nanmin(use_df['RMSE'])].to_numpy()[0]

                str = f'Generation {gen} ({nsims} Sims):\tMean = {mean_rmse} +/- {std_rmse}\tMedian RMSE = {med_rmse}\tBest RMSE = {best_rmse} (Sim {best_run})'
            f.write(f'{str}\n')
        f.write('\n')

        # Close the file
        f.close()
        print(f'File Saved: {filename}')

    """
    Class methods to plot the data
    """

    def average_profiles(self, factor=1, save=False):
        """
        Plot the average profile for each generation. Color by
        generation and shade in standard deviation

        factor: Number to multiply the standard deviation by
        save: Display the figure (False) or save it (True)
        """

        # Setup the figure
        fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
        cmap = plt.cm.cividis
        pp = self.profiles

        # Add a grid and an MHW line
        ax.grid(color='lightgrey', linewidth=0.5, zorder=0)
        ax.axhline(y=0.34, color='darkblue', linewidth=1, linestyle='--', zorder=2)
        ax.text(x=5, y=0.35, s='MHW', color='darkblue', zorder=1, **font)

        # Plot the final and initial field profile
        ax.plot(pp['X'], pp['Field Init'], color='red', linewidth=2, linestyle='--', label='Field$_{0}$', zorder=4)
        ax.plot(pp['X'], pp['Field Final'], color='red', linewidth=2, linestyle='-', label='Field$_{f}$', zorder=30)

        # Plot the initial model profile
        ax.plot(pp['X'], pp['Model Init'], color='black', linewidth=2, linestyle='--', label='Model$_{0}$', zorder=4)

        # Loop over the generations
        for gen in self.ngen:

            # Pull out the final profiles for the generation
            use_cols = [cc for cc in pp.columns if f'{gen}_' in cc]

            # Find the mean and sd
            df = pd.DataFrame()
            df['X'] = pp['X']
            df['Mean'] = pp[use_cols].mean(axis=1, skipna=True)
            df['Lo'] = df['Mean'] - (factor * pp[use_cols].std(axis=1, skipna=True))
            df['Hi'] = df['Mean'] + (factor * pp[use_cols].std(axis=1, skipna=True))

            # Shade in the standard deviation
            ax.fill_between(x=df['X'], y1=df['Hi'], y2=df['Lo'], facecolor=cmap(gen / self.ngen[-1]), alpha=0.25, zorder=gen + 4)

            # Plot the average profile
            ax.plot(df['X'], df['Mean'], color=cmap(gen / self.ngen[-1]), linewidth=2, zorder=gen + 6)

        # Add a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(f'Generation', labelpad=15, **font)
        cbar.set_ticks(np.linspace(0, 1, len(self.ngen)))
        cbar.set_ticklabels(self.ngen)

        # Add a legend
        ax.legend(loc='upper right', fancybox=False, edgecolor='black')

        # Set the X-Axis
        ax.set_xlim(left=0, right=self.adjusts['Right'])
        ax.set_xlabel('Cross-Shore Distance (m)', **font)

        # Set the Y-Axis
        ax.set_ylim(bottom=-1, top=self.adjusts['Top'])
        ax.set_ylabel('Elevation (m NAVD88)', **font)

        # Save and close the figure
        if self.special is None:
            title = os.path.join('Average Profiles', f'BGB{self.p} Average Profiles  ({self.t0} - {self.tf})')
        else:
            title = os.path.join('Average Profiles', f'BGB{self.p} Average Profiles  ({self.special})')
        save_and_close(fig, title, save)

    def best_profiles(self, save=False):
        """
        Plot the bestprofile for each generation. Color by
        generation and shade in standard deviation

        factor: Number to multiply the standard deviation by
        save: Display the figure (False) or save it (True)
        """

        # Setup the figure
        fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
        cmap = plt.cm.cividis
        pp = self.profiles

        # Add a grid and an MHW line
        ax.grid(color='lightgrey', linewidth=0.5, zorder=0)
        ax.axhline(y=0.34, color='darkblue', linewidth=1, linestyle='--', zorder=2)
        ax.text(x=5, y=0.35, s='MHW', color='darkblue', zorder=1, **font)

        # Plot the final and initial field profile
        ax.plot(pp['X'], pp['Field Init'], color='red', linewidth=2, linestyle='--', label='Field$_{0}$', zorder=4)
        ax.plot(pp['X'], pp['Field Final'], color='red', linewidth=2, linestyle='-', label='Field$_{f}$', zorder=30)

        # Plot the initial model profile
        ax.plot(pp['X'], pp['Model Init'], color='black', linewidth=2, linestyle='--', label='Model$_{0}$', zorder=4)

        # Loop over the generations
        for gen in self.ngen:

            # Find the profile with the lowest RMSE score
            temp = self.summary.loc[self.summary['Generation'] == gen]
            if temp['RMSE'].isnull().all():
                pass
            else:
                rmses = temp['RMSE'].to_numpy()
                ix = np.argwhere(rmses == np.nanmin(rmses))[0][0]
                best = ix + 1

                # Plot the average profile
                ax.plot(pp['X'], pp[f'{gen}_{best}'], color=cmap(gen / self.ngen[-1]), linewidth=2, zorder=gen + 6)

        # Add a colorbar
        divider = make_axes_locatable(ax)
        cax = divider.append_axes('right', size='5%', pad=0.05)
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
        cbar = fig.colorbar(sm, cax=cax)
        cbar.set_label(f'Generation', labelpad=15, **font)
        cbar.set_ticks(np.linspace(0, 1, len(self.ngen)))
        cbar.set_ticklabels(self.ngen)

        # Add a legend
        ax.legend(loc='upper right', fancybox=False, edgecolor='black', fontsize='small')

        # Set the X-Axis
        ax.set_xlim(left=0, right=self.adjusts['Right'])
        ax.set_xlabel('Cross-Shore Distance (m)', **font)

        # Set the Y-Axis
        ax.set_ylim(bottom=-1, top=self.adjusts['Top'])
        ax.set_ylabel('Elevation (m NAVD88)', **font)

        # Save and close the figure
        if self.special is None:
            title = os.path.join('Best Profiles', f'BGB{self.p} Best Profiles  ({self.t0} - {self.tf})')
        else:
            title = os.path.join('Best Profiles', f'BGB{self.p} Best Profiles  ({self.special})')
        save_and_close(fig, title, save)

    def boxplot(self, ylo=0, yhi=2, save=False):
        """
        Make a boxplot from results

        xunit: String with the X-Value unit (Default = None)
        yunit: String with the Y-Value unit (Default = None)
        save: Save the figure (True) or display it (False)
        """

        # Check for the neural network predictions file
        nn_filename = os.path.join('..', f'BGB{self.p} NN', f'BGB{self.p} Best ML Parameters.csv')
        if os.path.exists(nn_filename):
            nn_file = pd.read_csv(nn_filename)
            nn_rmse = nn_file['Expected RMSE'].mean()
        else:
            nn_rmse = None

        # Setup the figure
        fig, ax = plt.subplots(dpi=dpi, figsize=(inches, inches))
        values = ['RMSE', 'Dune RMSE', 'Beach RMSE']
        df = self.summary.melt(value_vars=values, id_vars='Generation')

        # Add a line for the NN RMSE
        if nn_rmse is not None:
            ax.axhline(y=nn_rmse, color='red', linewidth=1, linestyle='--', label='NN Prediction')

        # Plot the data
        sns.boxplot(x='Generation',
                    y='value',
                    hue='variable',
                    hue_order=values,
                    palette=['#1b9e77', '#d95f02', '#7570b3'],
                    data=df,
                    ax=ax)

        # Set a legend
        ax.legend(loc='upper right', title=None, fancybox=False, edgecolor='black', fontsize='x-small')

        # Set the X-Axis
        ax.set_xlabel('Generation', **font)

        # Set the Y-Axis
        ax.set_ylim(bottom=ylo, top=yhi)
        ax.set_ylabel('RMSE (m)', **font)

        # Save and close the figure
        if self.special is None:
            title = os.path.join('Boxplots', f'RMSE by Generation Boxplot for BGB{self.p} ({self.t0} - {self.tf})')
        else:
            title = os.path.join('Boxplots', f'RMSE by Generation Boxplot for BGB{self.p} ({self.special})')
        save_and_close(fig, title, save)

    def generation_overlays(self, save=False):
        """
        Plot the final profiles for all the runs in a generation
        and color them by the RMSE value. Loops over the generations
        and makes a figure for each one

        save: Bool to display the figure (False) or save it (True)
        """

        # Loop of the generations
        for gen in self.ngen:

            # Setup the figure
            fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
            cmap = plt.cm.viridis_r
            pp = self.profiles
            pop_size = self.summary['Simulation'].loc[self.summary['Generation'] == gen].max()
            rmses = self.summary['RMSE'].loc[self.summary['Generation'] == gen].to_numpy()

            # Add a grid and an MHW line
            ax.grid(color='lightgrey', linewidth=0.5, zorder=0)
            ax.axhline(y=0.34, color='darkblue', linewidth=1, linestyle='--', zorder=2)
            ax.text(x=5, y=0.35, s='MHW', color='darkblue', zorder=1, **font)

            # Plot the final and initial field profile
            ax.plot(pp['X'], pp['Field Init'], color='red', linewidth=2, linestyle='--', label='Field$_{0}$', zorder=4)
            ax.plot(pp['X'], pp['Field Final'], color='red', linewidth=2, linestyle='-', label='Field$_{f}$', zorder=20)

            # Plot the final model profiles
            for ix in range(1, pop_size + 1):
                if rmses[ix - 1] < 1:
                    p_color = cmap(rmses[ix - 1] / 1)
                else:
                    p_color = cmap(1 / 1)
                ax.plot(pp['X'], pp[f'{gen}_{ix}'], color=p_color, zorder=ix + 6)

            # Add a legend
            ax.legend(loc='upper right', title=f'Gen {gen}', fancybox=False, edgecolor='black')

            # Add a colorbar
            divider = make_axes_locatable(ax)
            cax = divider.append_axes('right', size='5%', pad=0.05)
            sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=1))
            cbar = fig.colorbar(sm, cax=cax)
            cbar.set_label(f'RMSE', labelpad=15, **font)

            # Set the X-Axis
            ax.set_xlim(left=0, right=self.adjusts['Right'])
            ax.set_xlabel('Cross-Shore Distance (m)', **font)

            # Set the Y-Axis
            ax.set_ylim(bottom=-1, top=self.adjusts['Top'])
            ax.set_ylabel('Elevation (m NAVD88)', **font)

            # Save and close the figure
            if self.special is None:
                title = os.path.join('Generational Overlays',
                                     f'BGB{self.p} Final Profiles for Generation {gen} ({self.t0} - {self.tf})')
            else:
                title = os.path.join('Generational Overlays',
                                     f'BGB{self.p} Final Profiles for Generation {gen} ({self.special})')
            save_and_close(fig, title, save)

    def params_plot(self, save=False):
        """
        Make a figure showing the parameter values used
        by generation for all parameters being tuned

        save: Bool to display the figure (False) or save it (True)
        """

        # Set a list of the columns with parameters
        p_cols = ['m', 'Cb', 'facAs', 'facSk', 'lsgrad', 'bedfric', 'wetslp', 'vegDensity']
        los = [0, 0, 0, 0, -0.1, 0, 0, 0]
        his = [15, 5, 1.0, 1.0, 0.1, 0.5, 1.0, 10]

        # Setup a figure
        fig, axes = plt.subplots(nrows=len(p_cols), figsize=(inches * 2, inches * 2), dpi=dpi,
                                 gridspec_kw=dict(wspace=0, hspace=0))

        # Loop through the axes and parameters
        for ax, col, default in zip(axes, p_cols, his):

            # Calculate the mean parameter value, the standard deviation in the parameter
            # value, and the mean RMSE value for that generation for each generation
            means, sds, scores = [], [], []
            for gen in self.ngen:
                if self.summary[col].loc[self.summary['Generation'] == gen].isnull().all():
                    means.append(None)
                    sds.append(None)
                    scores.append(None)
                else:
                    data = self.summary[col].loc[self.summary['Generation'] == gen]
                    means.append(np.nanmean(data))
                    sds.append(np.nanstd(data))
                    scores.append(np.nanmean(self.summary['RMSE'].loc[self.summary['Generation'] == gen]))

            # Add a horizontal line for the default value
            ax.axhline(y=default, color='red', linewidth=1, linestyle='--', zorder=2)

            # Plot the data
            ax.errorbar(x=self.ngen, y=means, yerr=sds, fmt='o', ecolor='darkgrey', elinewidth=2, zorder=4)
            ax.plot(self.ngen, means, color='black', linewidth=0.5, zorder=6)
            plot = ax.scatter(x=self.ngen, y=means, c=scores, cmap='viridis_r', edgecolors='black', linewidths=0.5,
                              vmin=0, vmax=1, zorder=8)

        # Setup the colorbar axis and add the bar
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes([0.92, 0.1, 0.02, 0.78])
        cbar = fig.colorbar(plot, orientation='vertical', cax=cbar_ax)
        cbar.set_label('RMSE$_{mean}$', **font)

        # Set the X-Axis
        axes[-1].set_xlabel('Generation', **font)
        axes[-1].set_xticks(self.ngen)
        [ax.set_xlabel('') for ax in axes[:-1]]

        # Set the Y-Axis
        [ax.set_ylim(bottom=lo, top=hi) for ax, lo, hi in zip(axes, los, his)]
        [ax.set_ylabel(col, **font) for ax, col in zip(axes, p_cols)]
        [ax.set_yticks([lo, (lo + hi)/2]) for ax, lo, hi in zip(axes, los, his)]

        # Set spines
        [ax.tick_params(left=False) for ax in axes[1:]]
        [ax.spines['bottom'].set_linewidth(0.5) for ax in axes[1:]]
        [ax.spines['top'].set_linewidth(0.5) for ax in axes[:-1]]

        # Save an close the figure
        if self.special is None:
            title = os.path.join('Param Figures', f'BGB{self.p} Parameter Values ({self.t0}-{self.tf})')
        else:
            title = os.path.join('Param Figures', f'BGB{self.p} Parameter Values ({self.special})')
        save_and_close(fig, title, save, tight=False)

    def rmse_plot(self, save=False):
        """
        Make a plot of the RMSE formulation for every simulation

        save: Bool to save (True) or display (False) the figure
        """

        for column in self.profiles.columns:
            if '_' in column:

                # Setup the figure
                fig, ((ax1, ax2), (ax3, ax4)) = \
                    plt.subplots(nrows=2, ncols=2, figsize=(inches * 2, inches * 2), dpi=dpi)
                axes = [ax1, ax2, ax3, ax4]
                pp = self.profiles[['X', 'Field Init', 'Field Final', 'Model Init', 'Mask', column]]
                gen, sim = column.split('_')
                pp['Z Diff'] = pp[column] - pp['Field Final']
                exes = pp['X'].loc[pp['Mask'] == 0]
                dune_exes = pp['X'].loc[pp['Mask'] == 1]
                beach_exes = pp['X'].loc[pp['Mask'] == 2]
                mask_pp = pp.loc[pp['Mask'] > 0]

                # Add grids
                [ax.grid(color='lightgrey', linewidth=0.5, zorder=0) for ax in axes]

                """
                Plot the gridded model and field overlay in the top left
                """

                # Add a box around the smooth zone, the dune, and the beach
                ax1.axvspan(exes.iloc[0], exes.iloc[-1], color='red', alpha=0.5, zorder=2)
                ax1.axvspan(dune_exes.iloc[0], dune_exes.iloc[-1], color='lightgreen', alpha=0.5, zorder=2)
                ax1.axvspan(beach_exes.iloc[0], beach_exes.iloc[-1], color='tan', alpha=0.5, zorder=2)

                # Add a line for MHW
                ax1.axhline(y=0.34, color='darkblue', linestyle='--', linewidth=2, zorder=4, label='MHW')

                # Plot the data
                ax1.plot(pp['X'], pp['Model Init'], color='r', linewidth=2, linestyle=':', zorder=6, label='Model$_{0}$')
                ax1.plot(pp['X'], pp['Field Init'], color='k', linewidth=2, linestyle=':', zorder=6, label='Field$_{0}$')
                ax1.plot(pp['X'], pp[column], color='r', linewidth=2, zorder=8, label='Model$_{f}$')
                ax1.plot(pp['X'], pp['Field Final'], color='k', linewidth=2, zorder=8, label='Field$_{f}$')

                # Add a legend
                ax1.legend(loc='upper right', fancybox=False, edgecolor='black', title=f'{gen}_{sim}')

                # Set the X-Axis
                ax1.set_xlim(left=0, right=self.adjusts['Right'])
                ax1.set_xlabel('Cross-Shore Distance (m)', **font)

                # Set the Y-Axis
                ax1.set_ylim(bottom=0, top=self.adjusts['Top'])
                ax1.set_ylabel('Elevation (m NAVD88)', **font)

                """
                Plot the final model versus field points in the top right
                """

                # Calculate a regression for the model versus field points and
                # add text to the plot with the results
                ex, why = mask_pp['Field Final'], mask_pp[column]
                slope, intercept, r_value, p_value, std_err = stats.linregress(ex, why)
                err_str = f'R2 = {np.around(r_value ** 2, decimals=3)}\np = {np.around(p_value, decimals=3)}'
                ax2.text(0.04, 0.80, err_str, ha='left', va='bottom', transform=ax2.transAxes, zorder=8, **font)

                # Add a 1:1 line
                ax2.plot([0, self.adjusts['Top']], [0, self.adjusts['Top']], color='black', linewidth=2, zorder=2)

                # Plot the data
                # ax2.scatter(x=mask_pp['Field Final'], y=mask_pp['Model Final'], c='r', edgecolors='k', linewidth=0.5, zorder=4)
                for ii, color in zip([1, 2], ['lightgreen', 'tan']):
                    ax2.scatter(x=mask_pp['Field Final'].loc[mask_pp['Mask'] == ii],
                                y=mask_pp[column].loc[mask_pp['Mask'] == ii],
                                c=color,
                                edgecolors='black',
                                linewidth=0.5,
                                zorder=4)

                # Set the X-Axis
                ax2.set_xlim(left=0, right=self.adjusts['Top'])
                ax2.set_xlabel('Field$_{f}$ (m NAVD88)', **font)

                # Set the X-Axis
                ax2.set_ylim(bottom=0, top=self.adjusts['Top'])
                ax2.set_ylabel('Model$_{f}$ (m NAVD88)', **font)

                """
                Plot the elevation difference between the final model
                versus the final field profiles in the lower left
                """

                # Add a box around the smooth zone, the dune, and the beach
                ax3.axvspan(exes.iloc[0], exes.iloc[-1], color='red', alpha=0.5, zorder=2)
                ax3.axvspan(dune_exes.iloc[0], dune_exes.iloc[-1], color='lightgreen', alpha=0.5, zorder=2)
                ax3.axvspan(beach_exes.iloc[0], beach_exes.iloc[-1], color='tan', alpha=0.5, zorder=2)

                # Plot the data
                ax3.plot(pp['X'], pp['Z Diff'], color='green', linewidth=2, zorder=4)

                # Add a zero line
                ax3.axhline(y=0, color='black', linewidth=2, linestyle='--', zorder=8)

                # Set the X-Axis
                ax3.set_xlim(left=0, right=self.adjusts['Right'])
                ax3.set_xlabel('Cross-Shore Distance (m NAVD88)', **font)

                # Set the Y-Axis
                y_limit = np.nanmax(np.abs(pp['Z Diff']))
                if np.isnan(np.ceil(y_limit)):
                    pass
                else:
                    ax3.set_ylim(bottom=-np.ceil(y_limit), top=np.ceil(y_limit))
                ax3.set_ylabel('$\Delta$Z (m, Model$_{f}$ - Field$_{f}$)', **font)

                """
                Plot a histogram of the final model minus final
                field points in the lower right
                """

                # Plot the data
                # ax4.hist(mask_pp['Z Diff'], edgecolor='black', linewidth=1, zorder=2)
                for ii, cc in zip([1, 2], ['lightgreen', 'tan']):
                    sns.kdeplot(data=pp['Z Diff'].loc[pp['Mask'] == ii].dropna(), legend=False, shade=True, color=cc, ax=ax4)

                # Add a zero-line
                ax4.axvline(x=0, color='darkgrey', linestyle='--', linewidth=2, zorder=2)

                # Calculate the errors and add a string
                # use_row = (self.summary['Generation'] == gen) & (self.summary['Simulation'] == sim)
                # err = np.around(self.summary['RMSE'].loc[use_row][0], decimals=3)
                err = rmse(df=pp, dec=3, column=column)
                dune_err = rmse(df=pp, dec=3, mask=1, column=column)
                beach_err = rmse(df=pp, dec=3, mask=2, column=column)
                err_str2 = f'RMSE = {err}\nDune RMSE = {dune_err}\nBeach RMSE = {beach_err}'
                ax4.text(0.04, 0.77, err_str2, ha='left', va='bottom', transform=ax4.transAxes, zorder=6, **font)

                # Set the X-Axis
                ax4.set_xlabel('$\Delta$Z (m, Model$_{f}$ - Field$_{f}$)', **font)

                # Set the Y-Axis
                ax4.set_ylabel('Count (-)', **font)

                """
                Make general changes to the figure
                """

                # Save and close
                if self.special is None:
                    title = os.path.join('RMSE Figures', f'BGB{self.p} ({self.t0} - {self.tf})', f'Error Figure {gen} {sim}')
                else:
                    title = os.path.join('RMSE Figures', f'BGB{self.p} ({self.special})', f'Error Figure {gen} {sim}')
                save_and_close(fig, title, save)


if __name__ == '__main__':
    test = WindsurfGAResult(p=22, t0=2016, tf=2017, dec=4)
    test.params_plot(save=False)
