"""
This script contains the code to create and run
a Windsurf instance

Michael Itzkin, 2/3/2021
"""

from scipy.interpolate import interp1d
from scipy.io import loadmat, savemat

import datetime as dt
import netCDF4 as nc
import pandas as pd
import numpy as np
import subprocess
import random
import shutil
import os

FIELD_DIR = os.path.join('..', '..', '..', 'Bogue Banks Field Data')


"""
Functions called by Windsurf class
"""


def adjusts(p, tf=2017):
    """
    The cross-shore positions of profiles needs to
    be adjusted. This function just returns the right
    values to adjust by
    """

    if p == 1 and (tf == 2017):
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
    elif p == 15 and (tf == 2017):
        field_tf_adjust = 6
        tail = 0
        fence = 25
        right = 60
        top = 6
    elif (p == 22) and (tf == 2017):
        field_tf_adjust = 2
        tail = 0
        fence = None
        right = 140
        top = 7

    return {
        'dField': field_tf_adjust,
        'Tail': tail,
        'Fence': fence,
        'Right': right,
        'Top': top,
    }


def copytree(src, dst, symlinks=False, ignore=None):
    """
    https://stackoverflow.com/a/12514470
    """
    if not os.path.exists(dst):
        os.makedirs(dst)
    for item in os.listdir(src):
        s = os.path.join(src, item)
        d = os.path.join(dst, item)
        if os.path.isdir(s):
            copytree(s, d, symlinks, ignore)
        else:
            if not os.path.exists(d) or os.stat(s).st_mtime - os.stat(d).st_mtime > 1:
                shutil.copy2(s, d)


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


def load_and_grid_profiles(p=22, t0=2016, tf=2017):
    """
    Load the field profile and model profiles and interpolate
    all onto the Windsurf X-Grid in order to analyze error metrics.

    Returns a DataFrame with the X-Grid, final and initial gridded
    profiles, and an array to mask out the "smooth" part of the profile

    p: Int with the profile number
    t0: Int with the year the hindcast starts at
    tf: Int with the year the hindcast ends at
    """

    # Adjust the profile to align them properly
    adjust_df = adjusts(p, tf)

    # Load the model and pull out the X-Grid and
    # initial profile. This is the same for every
    # time simulation so just pull from the first
    # for simplicity
    filename = os.path.join('windsurf.nc')
    data = nc.Dataset(filename)
    x_grid = -data['x'][:]
    initial_profile = data['zb'][:, 0]
    final_profile = data['zb'][:, -1]
    data.close()

    # Load the first and last field profile
    columns = ['X', 'Y', 'Z', 'Zsmooth', 'D', 'Lat', 'Lon']
    init_field_fname = os.path.join(FIELD_DIR, '{}'.format(t0), 'BGB{}_cross.csv'.format(p))
    final_field_fname = os.path.join(FIELD_DIR, '{}'.format(tf), 'BGB{}_cross.csv'.format(p))
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

    # Place the gridded profiles into a DataFrame
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


def rmse(df, dec=3, mask=1):
    """
    Calculate the RMSE for the model run

    df: DataFrame with the gridded profiles in it
    dec: Number of decimals to round to (Default = 3)
    mask: Int with the values to use to mask out the profile
    """

    # Cut out the smooth part of the profile
    df = df.loc[df['Mask'] == mask]

    # Calculate the RMSE
    y = df['Model Final']
    yhat = df['Field Final']
    err = y - yhat
    square_error = err**2
    mse = np.nanmean(square_error)
    root_mse = np.sqrt(mse)
    if dec is None:
        return root_mse
    else:
        return np.around(root_mse, decimals=dec)


"""
Windsurf class
"""


class Windsurf:
    """
    A class to setup, run, evaluate,
    and analyze a Windsurf simulation
    """

    def __init__(self, profile, local, t0=2016, tf=2017, defaults=True, _id=None):
        """
        Initialize the run

        profile: Int with the number of the profile being evaluated
        local: Bool to state if the algorithm is being ran locally (True) or on Longleaf (False)
        defaults: Bool to include default values (Default = True)
        _id: Include an id number to go to that run (good for testing)
        """

        # Get the current time as a string, this
        # will be used to set the folder name
        now = dt.datetime.now()
        now_string = now.strftime('%m%d%Y%H%M%S')

        # Load basic information
        self.local = local
        if self.local:
            WS_DIR = os.path.abspath('D:\Windsurf')
            self.BASE_DIR = os.path.join(WS_DIR, 'BGB{} Genetic Algorithm'.format(profile))
            self.coupling_code = os.path.join(WS_DIR, 'Coupling_Code')
        else:
            WS_DIR = os.path.join(os.path.sep, 'pine', 'scr', 'm', 'i', 'mitzkin', 'Windsurf')
            self.BASE_DIR = os.path.join(WS_DIR, 'BGB{}_GA'.format(profile))
            self.coupling_code = os.path.join(WS_DIR, 'Coupling_Code')
        ws_setup_fname = os.path.join(self.BASE_DIR, 'windsurf_setup.mat')
        self.params = loadmat(ws_setup_fname, squeeze_me=True, struct_as_record=False, mat_dtype=True)
        self.p = profile
        self.t0 = t0
        self.tf = tf
        if _id is None:
            self.id = '{}_{}'.format(now_string, random.randint(0, 1000000))
        else:
            self.id = _id

        # Set default parameter values
        if defaults:
            self.params['flow'].XB.bedfriction = 'manning'
            self.params['flow'].XB.bedfriccoef = 0.0228
            self.params['flow'].XB.nuh = 0.15

            self.params['project'].flag.Aeolis = 1.0
            self.params['project'].flag.CDM = 2.0
            self.params['project'].flag.XB = 1.0
            self.params['project'].flag.VegCode = 0.0
            self.params['project'].flag.fences = 0.0

            self.params['sed'].XB.form = 1.0
            self.params['sed'].XB.morfac = 10.0
            self.params['sed'].XB.bermslope = 0.0
            self.params['sed'].XB.sedcal = 1.0
            self.params['sed'].XB.lsgrad = 0.0
            self.params['sed'].XB.facAs = 0.1
            self.params['sed'].XB.facua = 0.1
            self.params['sed'].Aeolis.m = 0.5
            self.params['sed'].Aeolis.A = 0.085
            self.params['sed'].Aeolis.bi = 1.0

            self.params['veg'].CDM.beta = 200.0
            self.params['veg'].CDM.elevMin = 2.35
            self.params['veg'].CDM.maximumDensity = 10.0

            self.params['waves'].XB.alpha = 1.0
            self.params['waves'].XB.gamma = 0.50
            self.params['waves'].XB.dtheta = 360.0
            self.params['waves'].XB.thetamin = -180.0
            self.params['waves'].XB.thetamax = 180.0

        # Set the run directory
        self.params['project'].Directory = os.path.join(os.getcwd(), 'run{}'.format(self.id))

        # Set paths to the executables
        if self.local:
            gen_path = os.path.join('C:', 'Users', 'Geology', 'Documents', 'Model_Executables')
        else:
            gen_path = os.path.join('pine', 'scr', 'm', 'i', 'mitzkin', 'Windsurf', 'Model_Executables')
        self.params['project'].Aeolis.AeolisExecutable = os.path.join(gen_path, 'Aeolis', 'aeolis.exe')
        self.params['project'].CDM.CDMExecutable = os.path.join(gen_path, 'Coastal_Dune_Model', 'cdm.exe')
        self.params['project'].XB.XBExecutable = os.path.join(gen_path, 'XBeach', 'xbeach.exe')


    def evaluate_sim(self, dec=None):
        """
        Calculate an error metric for the model run

        dec: Number of decimals to run to. Use "none" to skip rounding
        """

        # Change into the model run folder
        os.chdir(self.params['project'].Directory)

        # Load and grid the field profile
        df = load_and_grid_profiles(self.p, self.t0, self.tf)

        # Calculate the rmse values
        full_rmse = rmse(df.loc[df['Mask'] >= 1], dec=dec)
        dune_rmse = rmse(df.loc[df['Mask'] == 1], dec=dec)
        beach_rmse = rmse(df.loc[df['Mask'] == 2], dec=dec)

        # Set a penalty for NaN values
        if np.isnan(full_rmse):
            full_rmse = 1000.0
        if np.isnan(dune_rmse):
            dune_rmse = 1000.0
        if np.isnan(beach_rmse):
            beach_rmse = 1000.0

        return full_rmse, dune_rmse, beach_rmse

    def print_params(self, decimals=4):
        """
        Function to print out the run parameters, could
        be useful for testing or debugging.
        """
        print('%----------------------------%')
        print('% Windsurf Run {self.id} Setup Parameters: %\n')

        print('\nFlow:')
        print('(XBeach) Bed Fric:\t\t{}'.format(self.params["flow"].XB.bedfriction))
        print('(XBeach) BFric Coeff:\t{}'.format(np.around(self.params["flow"].XB.bedfriccoef, decimals=decimals)))
        print('(XBeach) nuh:\t\t\t{}'.format(np.around(self.params["flow"].XB.nuh, decimals=decimals)))

        print('\nPaths:')
        print('(General) Path:\t{}'.format(self.params["project"].Directory))
        print('(Aeolis) Path:\t{}'.format(self.params["project"].Aeolis.AeolisExecutable))
        print('(CDM) Path:\t\t{}'.format(self.params["project"].CDM.CDMExecutable))
        print('(XB) Path:\t\t{}'.format(self.params["project"].XB.XBExecutable))

        print('\nProject:')
        print('(XBeach) Flag:\t\t{}'.format(np.around(self.params["project"].flag.XB, decimals=decimals)))
        print('(Aeolis) Flag:\t\t{}'.format(np.around(self.params["project"].flag.Aeolis, decimals=decimals)))
        print('(CDM) Flag:\t\t\t{}'.format(np.around(self.params["project"].flag.CDM, decimals=decimals)))
        print('(General) Veg:\t\t{}'.format(np.around(self.params["project"].flag.VegCode, decimals=decimals)))
        print('(General) Fences:\t{}'.format(np.around(self.params["project"].flag.fences, decimals=decimals)))

        print('\nSed:')
        print('(Aeolis) Cb:\t\t{}'.format(np.around(self.params["sed"].Aeolis.Cb, decimals=decimals)))
        print('(XBeach) facAs Lo:\t{}'.format(np.around(self.params["sed"].XB.facAsLo, decimals=decimals)))
        print('(XBeach) facAs Hi:\t{}'.format(np.around(self.params["sed"].XB.facAsHi, decimals=decimals)))
        print('(XBeach) facSk Lo:\t{}'.format(np.around(self.params["sed"].XB.facSkLo, decimals=decimals)))
        print('(XBeach) facSk Hi:\t{}'.format(np.around(self.params["sed"].XB.facSkHi, decimals=decimals)))
        print('(XBeach) lsgrad:\t{}'.format(np.around(self.params["sed"].XB.lsgrad, decimals=decimals)))
        print('(XBeach) Wet Slope:\t{}'.format(np.around(self.params["sed"].XB.wetslope, decimals=decimals)))
        print("(XBeach) Form:\t\t{}".format(np.around(self.params['sed'].XB.form, decimals=decimals)))
        print("(XBeach) Morfac:\t{}".format(np.around(self.params['sed'].XB.morfac, decimals=decimals)))
        print("(XBeach) Bermslope:\t{}".format(np.around(self.params['sed'].XB.bermslope, decimals=decimals)))
        print("(XBeach) Sedcal:\t{}".format(np.around(self.params['sed'].XB.sedcal, decimals=decimals)))
        print("(XBeach) Facua:\t\t{}".format(np.around(self.params['sed'].Aeolis.m, decimals=decimals)))
        print("(Aeolis) m:\t\t\t{}".format(np.around(self.params['sed'].Aeolis.m, decimals=decimals)))
        print("(Aeolis) A:\t\t\t{}".format(np.around(self.params['sed'].Aeolis.A, decimals=decimals)))
        print("(Aeolis) bi:\t\t{}".format(np.around(self.params['sed'].Aeolis.bi, decimals=decimals)))

        print('\nVeg:')
        print('(CDM) m:\t\t\t{}'.format(np.around(self.params["veg"].CDM.m, decimals=decimals)))
        print('(CDM) beta:\t\t\t{}'.format(np.around(self.params["veg"].CDM.beta, decimals=decimals)))
        print('(CDM) vegDensity:\t{}'.format(np.around(self.params["veg"].CDM.startingDensity, decimals=decimals)))
        print('(CDM) maxDensity:\t{}'.format(np.around(self.params["veg"].CDM.maximumDensity, decimals=decimals)))
        print('(CDM) elevMin:\t\t{}'.format(np.around(self.params["veg"].CDM.elevMin, decimals=decimals)))

        print('\nWaves:')
        print("(XBeach) Alpha:\t\t{}".format(np.around(self.params['waves'].XB.alpha, decimals=decimals)))
        print("(XBeach) Gamma:\t\t{}".format(np.around(self.params['waves'].XB.gamma, decimals=decimals)))
        print("(XBeach) dTheta:\t{}".format(np.around(self.params['waves'].XB.dtheta, decimals=decimals)))
        print("(XBeach) thetaMin:\t{}".format(np.around(self.params['waves'].XB.thetamin, decimals=decimals)))
        print("(XBeach) thetaMax:\t{}".format(np.around(self.params['waves'].XB.thetamax, decimals=decimals)))

    def run_windsurf(self):
        """
        Run a Windsurf instance
        """

        # Print out a message
        print('Starting Windsurf Run: {}'.format(self.id))

        # Copy runWindsurfFromPython.m into the run directory
        source = os.path.join('runWindsurfFromPython.m')
        destination = os.path.join(self.params['project'].Directory, 'runWindsurfFromPython.m')
        shutil.copyfile(source, destination)

        # Change into the run directory and run Windsurf
        os.chdir(self.params['project'].Directory)

        # Run MATLAB via a subprocess
        if self.local:
            subprocess.call('matlab -wait -nosplash -r runWindsurfFromPython', shell=True)
        else:
            subprocess.call('matlab -nosplash -nodesktop -singleCompThread -r runWindsurfFromPython', shell=True)

        # Change back to the project directory
        os.chdir(self.params['project'].Directory)

    def set_params(self, pp, save=True, newfolder=False):
        """
        Set Windsurf parameters from a supplied dict

        Right now this function is set to only take values only
        for specific parameters, will need to be modified to be more
        general if needed

        pp: Dict with parameter values to save
        save: Bool to save a new windsurf_setup.mat file
        newfolder: Bool to make a new folder to save the windsurf_setup.mat file into
        """

        # Set the vegetation parameters
        self.params['veg'].CDM.m = pp['m']
        self.params["veg"].CDM.startingDensity = pp['vegDensity']

        # Set the sed parameters
        self.params['sed'].Aeolis.Cb = pp['Cb']
        self.params['sed'].XB.facAsLo = pp['facAsLo']
        self.params['sed'].XB.facAsHi = pp['facAsHi']
        self.params['sed'].XB.facSkLo = pp['facSkLo']
        self.params['sed'].XB.facSkHi = pp['facSkHi']
        self.params['sed'].XB.lsgrad = pp['lsgrad']
        self.params["sed"].XB.wetslope = pp['wetslope']

        # Set the flow parameters
        self.params["flow"].XB.bedfriccoef = pp['bedfric']

        # Make a new folder if needed
        if newfolder:
            new_folder = os.path.join(self.BASE_DIR, 'run{}'.format(self.id))
            if not os.path.exists(new_folder):
                os.makedirs(new_folder)

        # Save the new params
        if save and newfolder:
            save_name = os.path.join(new_folder, 'windsurf_setup.mat')
            self.params['project'].Directory = os.path.join(self.BASE_DIR, 'run{}'.format(self.id))
            self.params['project'].couplingCode = self.coupling_code
            savemat(save_name, self.params)
        elif save:
            save_name = os.path.join('windsurf_setup.mat')
            savemat(save_name, self.params)
        else:
            pass


"""
Test out the code above
"""


def main():
    ws = Windsurf(profile=22, local=True, t0=2016, tf=2017, _id='4333538')
    print(ws.evaluate_sim(dec=None))


if __name__ == '__main__':
    main()
