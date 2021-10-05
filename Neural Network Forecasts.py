# -*- coding: utf-8 -*-
"""
Create a recurrent neural network to predict changes
in key dune metrics using output from Windsurf simulations
as the training data.


Created on Tue Apr  6 13:42:33 2021

@author: Geology
"""

from sklearn.preprocessing import MinMaxScaler, StandardScaler
from matplotlib.dates import DateFormatter
from tensorflow.keras import layers
import matplotlib.pyplot as plt
from tensorflow import keras
import datetime as dt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import os

# General settings
font = {'fontname': 'Arial', 'fontweight': 'normal', 'fontsize': 14}
dpi = 300
inches = 3.8
TEXT_DIR = os.path.join('..', 'Text Files')
FIELD_DIR = os.path.join('..', '..', 'Bogue Banks Field Data')
DATA_DIR = os.path.join('..', 'Data')
FIGURE_DIR = os.path.join('..', 'Figures')


"""
Functions to load and format data
"""


def load_field_morpho(p, metric):
    """
    Return a list of values from the field profiles for
    a given metric for every year of data
    
    p: Int with the profile number
    metric: String with the column to pull the vlaues from
    """
    
    # Set an empty list
    vals = []
    
    # loop through the years
    for year in [2016, 2017, 2018, 2019, 2020]:
    
        # Open the DataFrame
        fname = os.path.join(DATA_DIR, f'Morphometrics for BGB {year}.csv')
        df = pd.read_csv(fname)
        
        # Grab the value
        val = df[metric].loc[df['Profile'] == p].values[0]
        vals.append(val)
        
    return vals


"""
Functions to make figures
"""


def envrionmental_comparison_plot(df, ix, save=False):
    """
    Make a series of scatterplots comparing the
    environmental forcing data for the training
    year (2016-2017) to the testing year (2017-2020)
    
    df: DataFrame with envrionmental data
    ix: Index to split the DataFrame into 2016-2017 and 2017-2020
    save: Bool to display the figure (False) or save and close it (True)
    """
    
    # Add a "group" column to the dataframe to identify
    # if it is in the 2016-2017 group or the 2017-2020 group
    df['Group'] = 2             # 2017-2020
    df['Group'].iloc[ix] = 1    # 2016-2017
    
    #Setup the figure
    fig, axes = plt.subplots(ncols=6, nrows=6,
                             sharex='col', dpi=dpi,
                             figsize=(inches * 2, inches),
                             gridspec_kw={'wspace': 0, 'hspace': 0})
    colors = ['black', 'red']
    groups = [2, 1]
    cols = ['Tide', 'Hs', 'Tp', 'WD', 'WindDir', 'WindSpeed']
    labels = ['Tide (m)', 'Hs (m)', 'Tp (s)', 'WD ($^{o}$)', 'WindDir ($^{o}$)', 'U (m/s)']
    axes = axes.ravel().tolist()
    kde_ax = [0, 7, 14, 21, 28, 35]
    use_ax = [6, 12, 13, 18, 19, 20, 24, 25, 26, 27, 30, 31, 32, 33, 34]
    no_ax = [1, 2, 3, 4, 5, 8, 9, 10, 11, 15, 16, 17, 22, 23, 29]
    left_ax = [0, 6, 12, 18, 24, 30]
    bottom_ax = [30, 31, 32, 33, 34, 35]
    
    # Remove the un-necessary axes
    # Hide un-necessary axes
    [axes[ii].axis('off') for ii in no_ax]
    
    # Loop through the KDE Axes
    for ii, param in zip(kde_ax, cols):
        
        # Plot the KDE plot
        sns.kdeplot(df[param], ax=axes[ii], palette=['red', 'black'], hue=df['Group'], legend=False)
        
        # Count the out of sample values
        lo = df[param].loc[df['Group'] == 1].min()
        hi = df[param].loc[df['Group'] == 1].max()
        vals = df[param].loc[df['Group'] == 2].values
        n_oos = sum((vals < lo) | (vals > hi))
        pct_oos = (n_oos / len(vals)) * 100
        axes[ii].set_title(f'OOS: {n_oos} ({np.around(pct_oos, decimals=2)}%)', fontsize=8)
        
    # Plot the Hs v. X row
    for c, g in zip(colors, groups):
        axes[6].scatter(x='Tide', y='Hs', c=c, s=10, edgecolors='black', linewidth=0.25, data=df.loc[df['Group'] == g])
        
    # Plot the Tp v. X row
    for ii, ex in zip([12, 13], ['Tide', 'Hs']):
            for c, g in zip(colors, groups):
                    axes[ii].scatter(x=ex, y='Tp', c=c, s=10,
                        edgecolors='black', linewidth=0.25,
                        data=df.loc[df['Group'] == g])
                    
    # Plot the Wd v. X row
    for ii, ex in zip([18, 19, 20], ['Tide', 'Hs', 'Tp']):
            for c, g in zip(colors, groups):
                    axes[ii].scatter(x=ex, y='WD', c=c, s=10,
                        edgecolors='black', linewidth=0.25,
                        data=df.loc[df['Group'] == g])
                    
    # Plot the WindDir v. X row
    for ii, ex in zip([24, 25, 26, 27], ['Tide', 'Hs', 'Tp', 'WD']):
            for c, g in zip(colors, groups):
                    axes[ii].scatter(x=ex, y='WindDir', c=c, s=10,
                        edgecolors='black', linewidth=0.25,
                        data=df.loc[df['Group'] == g])
                    
    # Plot the WindSpeed v. X row
    for ii, ex in zip([30, 31, 32, 33, 34], ['Tide', 'Hs', 'Tp', 'WD', 'WindDir']):
            for c, g in zip(colors, groups):
                    axes[ii].scatter(x=ex, y='WindSpeed', c=c, s=10,
                        edgecolors='black', linewidth=0.25,
                        data=df.loc[df['Group'] == g])
    
    # Set the X-Axes
    [axes[ii].set_xlabel(col, fontsize=8) for ii, col in zip(bottom_ax, labels)]

    # Set the Y-Axes
    [axes[ii].set_ylabel(col, fontsize=8) for ii, col in zip(left_ax, labels)]
    [axes[ii].set_ylabel('') for ii in kde_ax[1:]]
    
    # Add a text label for the years
    for y, s, c in zip([0.75, 0.50], ['2016-2017', '2017-2020'], ['red', 'black']):
        ax = axes[35]
        ax.text(x=0.4, y=y, s=s, color=c, zorder=10, transform=ax.transAxes, fontsize=8)

    # Clear the ticks and labels for non "edge" subplots
    for ii in range(0, len(axes)):
        if ii not in left_ax and ii not in bottom_ax:
            axes[ii].set_yticks([])
    [axes[ii].set_yticks([]) for ii in bottom_ax[1:]]
    
    # Save and the close the figure
    title = 'Environmental Forcings Comparison'
    save_and_close(fig, title, save)
    

def forecast_plot(y_pred, y_ws, p, metric, fmetric, ylab, yerr=None,
                  ylim=(None, None), lo=None, mid=None, hi=None, save=False):
    """
    Plot the Windsurf and LSTM model predictions
    along with the field values for all years 2016-2020
    
    y_pred: Values outputted from the LSTM model
    y_ws: Values outputted from Windsurf
    p: Int with the profile number being looked at
    metric: String with the metric being looked at
    ylab: String with the label for the Y-Axis
    yerr: Value to use for the errorbars
    ylim: Tuple with the upper and lower Y-Values
    lo: Predictions with lo SLR rate
    mid: Predictions with mid SLR rate
    hi: Predictions with hi SLR rate
    save: Bool to display the figure (False) or save and close it (True)
    """
    
    # Set the years and field values
    # THIS WILL HAVE TO BE IMPROVED!!!
    times = pd.date_range(start='10/16/2016', freq='1H', periods=len(y_pred))
    field_times = [
        dt.datetime(year=2016, month=10, day=16, hour=00),
        dt.datetime(year=2017, month=10, day=12, hour=00),
        dt.datetime(year=2018, month=10, day=12, hour=00),
        dt.datetime(year=2019, month=11, day=10, hour=00),
        dt.datetime(year=2020, month=12, day=5, hour=00)
    ]
    field = load_field_morpho(p, fmetric)
    
    # Setup the figure
    fig, ax = plt.subplots(dpi=300)
    
    # Add a grid
    ax.grid(color='lightgrey', linewidth=0.5, zorder=0)
    
    # Plot the LSTM predictions
    ax.plot(times, y_pred, color='black', label='LSTM Prediction', zorder=4)
    if lo is not None and hi is not None:
        ax.fill_between(x=times, y1=hi, y2=lo, facecolor='lightgrey', label='Std.', zorder=2)
    
    # Plot the Windsurf prediction
    ax.plot(times[:len(y_ws)], y_ws, color='red', label='Windsurf', zorder=20)
    
    # Plot the field values
    ax.errorbar(x=field_times,
                y=field,
                linewidth=0,
                fmt='o', 
                elinewidth=1,
                yerr=yerr,
                color='blue',
                label='Field',
                zorder=6)
    
    # Add a legend
    if 'Toe' in fmetric:
        ax.legend(loc='lower left', fancybox=False, edgecolor='black')
    else:
        ax.legend(loc='upper left', fancybox=False, edgecolor='black')
    
    # Set the X-Axis
    fmt = DateFormatter('%b-%Y')
    ax.xaxis.set_major_formatter(fmt)
    fig.autofmt_xdate(bottom=0.2, rotation=30, ha='right')
    ax.set_xlim(left=dt.datetime(year=2016, month=10, day=1, hour=00),
                right=dt.datetime(year=2021, month=1, day=1, hour=00))
    
    # Set the Y-Axis
    ax.set_ylim(ylim)
    ax.set_ylabel(ylab, **font)
    
    # Save and the close the figure
    title = f'BGB{p} {metric} Forecast'
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
    if save:
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
Functions to work with the neural network
"""

def create_dataset(X, y, time_steps=1):
    Xs, ys = [], []
    for i in range(len(X) - time_steps):
        v = X[i:(i + time_steps)]
        Xs.append(v)
        ys.append(y[i + time_steps])
    return np.array(Xs), np.array(ys)


def make_model(X, y, metric, p, epochs=100, batch_size=10, new=False):
    """
    Compile and train the model or just load
    a pre-trained model
    
    X: X-values for the training data
    y: y-values for the training data
    metric: String with the metric beign trained on
    p: Int with the model number
    epochs: Int with the number of training epochs
    batch_size: Int with the batch size for training
    new: Bool to make a new model or not (Default: False)
    """
    
    # Set the saved model name
    fname = os.path.join(DATA_DIR, f'BGB{p}', f'BGB{p} {metric} LSTM')
    
    if new:
    
        # Make the LSTM model
        model = keras.Sequential([
                layers.LSTM(units=128, input_shape=(X.shape[1], X.shape[2])),
                layers.Dense(units=1, activation='linear')
                ])
        model.compile(loss='mean_squared_error',
                      optimizer='Adam')
        
        # Train the LSTM model
        history = model.fit(
            X, y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=0.1,
            verbose=1,
            shuffle=False
        )
        
        # Save the model
        model.save(fname)
        
    else:
        
        # Load the model
        model = keras.models.load_model(fname)
    
    return model


def make_predictions(model, df, cols, xsc, ysc, tsteps, slr=0):
    """
    Make predictions using the trained LSTM model
    
    model: Trained LSTM neural network model
    df: DataFrame with the environmental forcings
    cols: List of strings with the columns to use
    xsc: StandardScaler() object for the X-Values
    ysc: StandardScaler() object for the Y-Values
    tsteps: Int with the number of time steps to pass to create_dataset()
    slr: Number with the linear SLR rate to apply (Default = 0)
    """
    
    # Make the predictions
    X_full = df[cols].to_numpy()
    X_full = xsc.transform(X_full)
    X_full, _ = create_dataset(X_full, X_full, tsteps)
    y_pred = model.predict(X_full)
    y_pred = ysc.inverse_transform(y_pred)
    
    return y_pred


"""
Run the forecasts
"""


def main():
    """
    Run the forecasts
    """
    
    # Set the values to use
    profile = 22
    time_steps = 48
    epochs = 25
    batch_size = 5000
    metric = 'Y Toe'
    fmetric = 'yToe'
    new_lstm = False
    ylabel = 'D$_{low}$ (m NAVD88)'
    y_err = 0.02
    fig_lim = (2.0, 2.50)
    save_fig = True
    
    
    # Load the environmental forcings and morphometrics
    env_df = pd.read_csv(os.path.join(DATA_DIR, '2016 to 2020 CALO Environmental Data.csv'))
    morpho_df = pd.read_csv(os.path.join(DATA_DIR, f'BGB{profile}', 'windsurf morphometrics.csv'))
    
    # Make a plot of the environmental forcings for
    # 2016-2017 and 2017-2020
    envrionmental_comparison_plot(env_df, morpho_df.index, save=True)
    
    # Cut off the environmental forcings to match the length
    # of the morphometrics and horizontally concatenate the 
    # two DataFrames
    df = pd.concat([env_df.iloc[morpho_df.index], morpho_df], axis=1)
    
    # Make "difference" columns for key metrics
    for col in df.columns:
        if 'Times' in col or'Label' in col:
            pass
        else:
            df[f'Delta {col}'] = df[col].diff().fillna(0)
            
            if col in env_df.columns:
                env_df[f'Delta {col}'] = env_df[col].diff().fillna(0)
    
    # Pull out the columns needed for the X and Y values
    y = df[f'Delta {metric}'].to_numpy().reshape(-1, 1)
#    x_cols = ['Delta Hs', 'Delta Tp', 'Delta WD',
#              'Delta Tide', 'Delta WindDir', 'Delta WindSpeed']
#    x_cols = ['Hs', 'Tp', 'WD', 'Tide', 'WindDir', 'WindSpeed', metric]
    x_cols = ['Delta Hs', 'Delta Tp', 'Delta WD',
              'Delta Tide', 'Delta WindDir', 'Delta WindSpeed',
              'Hs', 'Tp', 'WD', 'Tide', 'WindDir', 'WindSpeed']
    X = df[x_cols].to_numpy()
    
    # Scale the values
    xscaler = StandardScaler().fit(X)
    yscaler = StandardScaler().fit(y)
    X_scaled = xscaler.transform(X)
    y_scaled = yscaler.transform(y)
    
    # Train the LSTM model
    X_scaled, y_scaled = create_dataset(X_scaled, y_scaled, time_steps) 
    model = make_model(X_scaled, y_scaled, metric, profile,
                       epochs, batch_size, new=new_lstm)
    
    # Predict the Y-values the full time span in a loop to generate
    # a mean and deviation of the predictions instead of a single value
    pred_df = pd.DataFrame()
    for ii in range(10):
        
        # Add random noise to the forcing data
        eps = 0.05
        noise = np.random.normal(loc=0.0, scale=1.0, size=env_df.shape) * eps
        noise_df = pd.DataFrame(noise, columns=env_df.columns)
        noise_df = noise_df.add(env_df, fill_value=0)
        for col in noise_df.columns:
            if 'Delta' in col:
                pass
            else:
                noise_df[f'Delta {col}'] = noise_df[col].diff().fillna(0)
                
        pred_vals = make_predictions(model, noise_df, x_cols, xscaler, yscaler, time_steps)
        pred_df[f'{ii}'] = pred_vals.ravel().tolist()
        
    y_pred = make_predictions(model, env_df, x_cols, xscaler, yscaler, time_steps)
       
    # Calculate the mean and standard deviations of the predictions
    pred_cols = pred_df.columns
    pred_df['Mean'] = pred_df.mean(axis=1)
    pred_df['Std'] = pred_df[pred_cols].std(axis=1)
    pred_df['Lo'] = pred_df['Mean'] - pred_df['Std']
    pred_df['Hi'] = pred_df['Mean'] + pred_df['Std']
    pred_df['2 Lo'] = pred_df['Mean'] - (2 * pred_df['Std'])
    pred_df['2 Hi'] = pred_df['Mean'] + (2 * pred_df['Std'])
        
    # print(pred_df)
    
    # The neural network returns the change in values
    # so take a cumulative sum and add the initial value
    # to convert it back to the proper values
    mid_pred = np.nancumsum(pred_df['Mean'].values) + df[metric].iloc[0]
    lo_pred = np.nancumsum(pred_df['Lo'].values) + df[metric].iloc[0]
    hi_pred = np.nancumsum(pred_df['Hi'].values) + df[metric].iloc[0]
    y_pred = np.nancumsum(y_pred) + df[metric].iloc[0]
    y_ws = np.nancumsum(y) + df[metric].iloc[0]
    
    # Plot the predictions
    forecast_plot(y_pred, y_ws, profile, metric, fmetric, ylabel, y_err,
                  lo=lo_pred, mid=mid_pred, hi=hi_pred, ylim=fig_lim, save=save_fig)
    
    
if __name__ == '__main__':
    main()

