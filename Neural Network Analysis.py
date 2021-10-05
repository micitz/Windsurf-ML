"""
Analyze results from the runs that were tuned
using a neural network (Neural Network Calibration.py)

Michael Itzkin, 3/15/2021
"""

from scipy import io
from scipy.interpolate import interp1d

import matplotlib.pyplot as plt
import netCDF4 as nc
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
Create a Windsurf result class
"""


def adjusts(p, tf=2016):
    """
    The cross-shore positions of profiles needs to
    be adjusted. This function just returns the right
    values to adjust by
    """

    if p == 3 and (tf == 2017):
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
        fence = 25
        right = 60
        top = 6
    elif p == 15 and (tf == 2019):
        field_tf_adjust = -3
        tail = 10
        fence = 25
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

    for run in summ_df['Simulation']:
        if special is not None:
            USE_DIR = os.path.join('..', f'{special}')
        elif tf == 2017:
            USE_DIR = os.path.join('..', f'BGB{p} NN')
        pp = load_and_grid_profiles(DIR=os.path.join(USE_DIR, f'run{run}'), p=p, t0=t0, tf=tf)

        # The final model result is the only thing that differs between
        # simulations so eveything else can be pulled just from gen 1 run 1
        if run == 1:
            profiles_df['X'] = pp['X']
            profiles_df['Field Init'] = pp['Field Init']
            profiles_df['Field Final'] = pp['Field Final']
            profiles_df['Model Init'] = pp['Model Init']
            profiles_df['Mask'] = pp['Mask']
        profiles_df[f'Run {run}'] = pp['Model Final']

    return profiles_df


def make_summary_df(p, t0, tf, runs=24, dec=4, special=None):
    """
    Compile results from the GA simulations and store
    them in a DataFrame for analysis

    p: Int with the profile to analyze
    t0: Int with the initial year of analysis
    tf: Int with the final year of analysis
    runs: Int with the number of runs (Default = 24)
    dec: Int with the number of decimals to round to (Default = 4)
    """

    # Set empty lists to loop into. These will be used to
    # construct the DataFrame at the end
    nrun, m, cb, facas_lo, facas_hi, facsk_lo, facsk_hi, lsgrad, bedfric, wetslp, beta,\
    veg_density, rmses, drmses, brmses =\
        [], [], [], [], [], [], [], [], [], [], [], [], [], [], []

    # Set the path to the folder to work in
    if special is not None:
        USE_DIR = os.path.join('..', f'{special}')
    elif tf == 2017:
        USE_DIR = os.path.join('..', f'BGB{p} NN')

    # Loop through the folders
    for run in range(1, runs + 1):

        # Set a path to the run folder
        if special is None:
            filename = os.path.join(USE_DIR, f'run{run}', 'windsurf_setup.mat')
        else:
            filename = os.path.join(USE_DIR, f'BGB{p}' f'run{run}', 'windsurf_setup.mat')

        # Load the parameters sued
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

        # Calculate the RMSE value
        if special is None:
            temp_dir = os.path.join(USE_DIR, f'run{run}')
        else:
            temp_dir = os.path.join(USE_DIR, f'BGB{p}', f'run{run}')

        df = load_and_grid_profiles(temp_dir, p, t0, tf)
        rmses.append(rmse(df.loc[df['Mask'] >= 1], dec=dec))
        drmses.append(rmse(df.loc[df['Mask'] == 1], dec=dec))
        brmses.append(rmse(df.loc[df['Mask'] == 2], dec=dec))

        # Update the lists
        nrun.append(run)

    return pd.DataFrame.from_dict({
        'Simulation': nrun,
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


class WindsurfResult():

    def __init__(self, p, t0=2016, tf=2017, runs=24, dec=4, special=None):
        """
        Initialize the results object

        p: Int with the profile number being worked on
        t0: Int with the start year of the simulation
        tf: Int with the final year of the simulation (Default = 2016)
        runs: Number of runs performed (Default = 24_
        dec: Int with the number of decimal places to round to (Default = 4)
        special: String with the folder name of a combo calibration run (Default = None)
        """

        # Initialize input arguments
        self.p = p
        self.t0 = t0
        self.tf = tf
        self.dec = dec
        self.runs = runs
        self.adjusts = adjusts(self.p, self.tf)
        if special is None:
            self.special = None
        else:
            self.special = f'BGB15 and BGB22 NN'

        # Make a summary DataFrame with run parameters and error metrics
        self.summary = make_summary_df(self.p, self.t0, self.tf, self.runs, self.dec, self.special)

        # Make a profiles Dataframe with the gridded model and field profiles
        self.profiles = make_profiles_df(self.summary, self.p, self.t0, self.tf, self.special)


"""
Functions to make figures
"""


def nn_overlays(ws, save=False):
    """
    Plot all the neural network final profiles over the initial
    and final field profiles. Highlight the "best" neural network
    result

    ws: WindsurfResult object to work with
    save: Display the figure (False) or save it (True)
    """

    # Setup the figure
    fig, ax = plt.subplots(dpi=dpi, figsize=(inches, inches))
    best = ws.summary['Simulation'].loc[ws.summary['RMSE'] == np.nanmin(ws.summary['RMSE'])].values[0]

    # Add a grid and a line for MHW
    ax.grid(color='lightgrey', linewidth=0.5, zorder=0)
    ax.axhline(y=0.34, color='darkblue', linewidth=2, linestyle='--', zorder=2, label='MHW')

    # Add a line for the fence if necessary
    if ws.adjusts['Fence'] is None:
        pass
    else:
        ax.axvline(x=ws.adjusts['Fence'], color='sienna', linewidth=1, zorder=4, label='Fence')

    # Plot the initial and final field profiles
    ax.plot(ws.profiles['X'], ws.profiles['Field Init'], color='black', linewidth=2, linestyle='--', zorder=4, label='Field$_{0}$')
    ax.plot(ws.profiles['X'], ws.profiles['Field Final'], color='black', linewidth=2, linestyle='-', zorder=30, label='Field$_{f}$')

    # Plot the initial model profile and best model profile
    ax.plot(ws.profiles['X'], ws.profiles['Model Init'], color='red', linewidth=2, linestyle='--', zorder=4, label='Model$_{0}$')
    ax.plot(ws.profiles['X'], ws.profiles[f'Run {best}'], color='red', linewidth=2, linestyle='-', zorder=40, label='Model$_{f}$')

    # Plot all the other model results
    for ii in range(1, ws.runs+1):
        ax.plot(ws.profiles['X'], ws.profiles[f'Run {ii}'],
                color='darkgray', linewidth=1, linestyle='-', zorder=6 + ii)

    # Add a legend
    ax.legend(loc='upper right', fancybox=False, edgecolor='black')

    # Set the X-Axis
    ax.set_xlim(left=0, right=ws.adjusts['Right'])
    ax.set_xlabel('Cross-Shore Distance (m)', **font)

    # Set the Y-Axis
    ax.set_ylim(bottom=-1, top=ws.adjusts['Top'])
    ax.set_ylabel('Elevation (m NAVD88)', **font)

    # Save and close the figure
    title = f'BGB{ws.p} Neural Network Final Profiles'
    save_and_close(fig, title, save)


def save_and_close(fig, title, save):
    """
    Give the figure a tight and transparent background
    then close it

    fig: Figure object
    title: String with the title of the figure
    save: Save the figure (True) of display it (False)
    """

    # Set a tight layout and a transparent background
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
Run the analysis
"""


def main():
    """
    Run the analysis
    """

    # Load the results
    bgb22 = WindsurfResult(p=22)
    bgb15 = WindsurfResult(p=15)

    # Plot overlays of the results
    [nn_overlays(df, save=True) for df in [bgb15, bgb22]]


if __name__ == '__main__':
    main()
