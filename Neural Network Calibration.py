"""
Calibrate a hindcast using a neural network model
and a genetic algorithm to predict potential ideal
parameterizations

1. Compile all of the parameters and RMSE values from every
    BGB22 simulation into a DataFrame
2. Train a neural network on the DataFrame
3. Find optimal parameter values by running a genetic
    algorithm through the neural network
4. Run Windsurf using the optimal parameter values from step 3

Michael Itzkin, 3/8/2021
"""

from tensorflow import keras
from tensorflow.keras import layers

from deap import base, creator, tools, algorithms
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import random
import time
import os

# General settings
font = {'fontname': 'Arial', 'fontweight': 'normal', 'fontsize': 14}
dpi = 300
inches = 3.8
TEXT_DIR = os.path.join('..', 'Text Files')
FIELD_DIR = os.path.join('..', '..', 'Bogue Banks Field Data')


"""
Functions to load and prep data
"""


def make_params(p):
    """
    Load the tables with the run parameters
    and RMSE values into a single DataFrame
    
    p: Int with the profile number
    """

    # Make an empty DataFrame
    df = pd.DataFrame()

    # Set a list of columns to use
    cols = ['m', 'Cb', 'facAs', 'facSk', 'lsgrad', 'wetslp', 'vegDensity', 'RMSE']

    # Set paths to the tables
    DATA_DIR = os.path.join('..', 'Text Files')
    single_calibration_fname = os.path.join(DATA_DIR, f'Calibration Parameters (Profile {p}).csv')
    ga_calibration_fname = os.path.join(DATA_DIR, 'Master Genetic Algorithm Parameters List.csv')

    # Load the tables
    single_df = pd.read_csv(single_calibration_fname, delimiter=',', header=0)
    ga_df = pd.read_csv(ga_calibration_fname, delimiter=',', header=0)

    # Put the values from the single calibrations into the DataFrame
    df[cols] = single_df[cols]
    df['bedfric'] = single_df['bedfriccoef']

    # Put the bedfric column into the list of columns
    cols.append('bedfric')

    # Put the values from the individual genetic algorithm into
    # the main DataFrame
    use_df = ga_df.loc[(ga_df['Profile'] == p) & (ga_df['Long'] == False)]
    df = pd.concat([df, use_df[cols]])

    # Put the values from the combined genetic algorithm into
    # the main DataFrame
    if p == 15 or p == 22:
        combined_ga_calibration_fname = os.path.join(DATA_DIR, 'All Combined Genetic Algorithm Parameters.csv')
        combined_df = pd.read_csv(combined_ga_calibration_fname, delimiter=',', header=0)
        combined_df.rename(columns={f'BGB{p} RMSE': 'RMSE'}, inplace=True)
        df = pd.concat([df, combined_df[cols]])

    # Early simulations where vegetation density
    # wasn't being tested had a value of 1 so fill the
    # empty rows of this column with 1s
    df['vegDensity'].fillna(1, inplace=True)

    # Drop Nans and reset the index
    df.dropna(inplace=True)
    df.reset_index(drop=True, inplace=True)

    return df


def set_bounds(p, eps=0.05):
    """
    Return a dict of parameter bounds for
    a given profile.
    
    Better predictions can be made with some prior knowledge, so modify
    this function to return more narrow bounds for profiles where parameter
    tests have already been ran
    
    p: Int with the profile number
    eps: Amount to vary the custom bounds by to set a high and low (Default = 0.5)
    """
    
    # Set the generic dict
    bounds = {'m': [0.00, 15.00],
              'Cb': [0.00, 3.00],
              'facAs': [0.00, 1.00],
              'facSk': [0.00, 1.00],
              'lsgrad': [-0.1, 0.1],
              'bedfric': [0.00, 0.50],
              'wetslp': [0.00, 1.00],
              'vegDensity': [0.00, 10.00]}
    
    # Set specific bounds for profiles where you know something already
    # about the parameters that might be needed
    if p == 3:
        bounds['vegDensity'] = [0.1, 2.0]
    elif p == 15:
        bounds['m'] = [0.1, 2.0]
        bounds['facAs'] = [0.10, 0.60]
        bounds['facSk'] = [0.10, 0.60]
        bounds['lsgrad'] = 0.055
        bounds['bedfric'] = [0.1, 0.2]
        bounds['wetslp'] = [0.20, 0.70]
    elif p == 22:
        bounds['m'] = [5, 15]
        bounds['Cb'] = [0.05, 0.10]
        bounds['facAs'] = [0.65, 0.75]
        bounds['facSk'] = [0.60, 0.70]
        bounds['lsgrad'] = 0.012
        bounds['bedfric'] = 0.024
        bounds['wetslp'] = [0.40, 0.50]
        bounds['vegDensity'] = [0.1, 2.0]
        
    # If a single value has been found for a given bound, turn it
    # into an upper and lower bound by multiplying it by eps
    for k, v in bounds.items():
        if not isinstance(v, list):
            bounds[k] = [(v - (v * eps)), (v + (v * eps))]

    return bounds


"""
Functions to make figures and text files
"""


def ga_loss_plot(log, min_val, profile, save=False):
    """
    Make a plot of the genetic algorithm loss
    function over each generation with the mean
    and 1 sd
    
    log: Logbook object from the algorithm
    min_val: Lowest RMSE score from the actual runs
    """
    
    # Pull out the values needed
    gen, avg, sd, low, high = log.select('gen', 'avg', 'std', 'min', 'max')
    hi = np.array(avg) + np.array(sd)
    lo = np.array(avg) - np.array(sd)
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    
    # Add a grid
    ax.grid(color='lightgrey', linewidth=0.5, zorder=0)
    
    # Add a horizontal line with the best value
    # ax.axhline(y=min_val, color='red', linewidth=2, linestyle='--', zorder=8, label='Best')
    
    # Plot the loss function
    ax.fill_between(x=gen, y1=hi, y2=lo, facecolor='blue', alpha=0.5, zorder=2, label='Std. Dev.')
    ax.plot(gen, avg, color='blue', zorder=4, label='Mean')
    ax.plot(gen, low, color='blue', linestyle='--', zorder=6, label='Min')
    
    # Add a legend
    ax.legend(loc='upper right', fancybox=False, edgecolor='black')
    
    # Set the X-Axis
    ax.set_xlim(left=0, right=np.nanmax(gen))
    ax.set_xlabel('Generation', **font)
    
    # Set the Y-Axis
    ax.set_ylim(bottom=0, top=np.ceil(np.nanmax(avg + sd)))
    ax.set_ylabel('RMSE (m)', **font)
    
    # Save and close the figure
    title = 'Genetic Algorithm RMSE Values'
    save_and_close(fig, title, profile, save)


def nn_loss_plot(h, profile, save=False):
    """
    Plot the training and validation loss for
    the neural network versus the epochs

    h: tf.keras History object
    profile: Int with the profile number
    save: Bool to display (False) or save (True) the figure
    """
    
    # Pull out the values to plot
    t_loss = h.history['loss']
    v_loss = h.history['val_loss']
    epochs = range(1, len(t_loss)+1)

    # Setup the figure
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    
    # Add a grid
    ax.grid(color='lightgrey', linewidth=0.5, zorder=0)
    
    # Plot the loss function
    ax.plot(epochs, t_loss, color='blue', zorder=4, label='Training Loss')
    ax.plot(epochs, v_loss, color='red', zorder=2, label='Validation Loss')
    
    # Add a legend
    ax.legend(loc='upper right', fancybox=False, edgecolor='black')
    
    # Set the X-Axis
    ax.set_xlim(left=1, right=max(epochs))
    ax.set_xlabel('Epochs', **font)
    
    # Set the Y-Axis
    ax.set_ylim(bottom=0, top=0.1)
    ax.set_ylabel('Loss', **font)
    
    # Save and close the figure
    title = 'Neural Network Loss v Epoch'
    save_and_close(fig, title, profile, save)


def nn_pred_obs_scatter(x_test, y_test, x_train, y_train, x_val, y_val, model, profile, save=False):
    """
    Make a scatter plot of RMSE predictions from
    Windsurf v. RMSE predictions from the neural
    network. Calculate the R2 value
    
    y_test: RMSE values from Windsurf used for testing data
    y_train: RMSE values from Windsurf used for training data
    y_val: RMSE values from Windsurf used for validation data
    model: Trained neural network
    profile: Int with the profile number
    save: Bool to display (False) or save (True) the figure
    """
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    whys = [y_train, y_val, y_test]
    exes = [model.predict(x_train), model.predict(x_val), model.predict(x_test)]
    colors = ['blue', 'green', 'red']
    labs = ['Training', 'Validation', 'Testing']
    
    # Perform the regression on the testing data
    X = model.predict(x_test)
    reg = LinearRegression(fit_intercept=False).fit(X=X.reshape(-1, 1), y=y_test.reshape(-1, 1))
    r2 = reg.score(X.reshape(-1, 1), y_test.reshape(-1, 1))
    
    # Add a grid
    ax.grid(color='lightgrey', linewidth=0.5, zorder=0)
    
    # Add a 1:1 line
    ax.plot([0, 2], [0, 2], color='black', linewidth=1, zorder=2)
    
    # Plot the Windsurf v. NN values
    for x, y, c, l in zip(exes, whys, colors, labs):
        ax.scatter(x=x, y=y, facecolors=c, edgecolors='k', zorder=4, label=l)
    
    # Add text with the R2 value
    ax.text(x=0.05, y=1.85, s=f'R2 = {np.around(r2, decimals=3)}', **font)
    
    # Add a legend
    ax.legend(loc='lower right', fancybox=False, edgecolor='black')
    
    # Set the X-Axis
    ax.set_xlim(left=0, right=2)
    ax.set_xticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_xlabel('RMSE$_{Neural Network}$ (m)', **font)
    
    # Set the Y-Axis
    ax.set_ylim(bottom=0, top=2)
    ax.set_yticks([0, 0.5, 1.0, 1.5, 2.0])
    ax.set_ylabel('RMSE$_{Windsurf}$ (m)', **font)
    
    # Save and close the figure
    title = 'Windsurf v NN RMSE Values'
    save_and_close(fig, title, profile, save)


def parameter_pairplot(df, p, profile, save=False):
    """
    Make a pairplot of the parameter values colored
    by RMSE value

    df: DataFrame with parameter values and RMSE scores
    p: Dict with parameter values and bounds
    profile: Int with the profile number
    save: Bool to display (False) or save (True) the figure
    """

    # Setup the figure
    fig, ax = plt.subplots(figsize=(inches * 2, inches * 2), dpi=dpi)
    pd.plotting.scatter_matrix(df[p.keys()],
                               c=df['RMSE'],
                               cmap='Reds_r',
                               vmin=0, vmax=1,
                               ax=ax)

    # Save and close the figure
    title = f'BGB{profile} Parameter Pairs'
    save_and_close(fig, title, profile, save)


def params_v_rmse(df, p, profile, save=False):
    """
    Make a multi-panel scatter plot of 
    RMSE versus the parameter values.
    
    df: DataFrame with parameter values and RMSE scores
    p: Dict with parameter values and bounds
    profile: Int with the profile number
    save: Bool to display (False) or save (True) the figure
    """
    
    # Setup the figure
    fig, ((ax1, ax2, ax3, ax4), (ax5, ax6, ax7, ax8)) =\
        plt.subplots(nrows=2, ncols=4, sharey='all', dpi=dpi,
                     figsize=(inches * 2, inches))
    axes = [ax1, ax2, ax3, ax4, ax5, ax6, ax7, ax8]
    lefts = [0, 0, 0, 0, -0.1, 0, 0, 0]
    rights = [20, 3, 1, 1, 0.1, 0.6, 1, 10]
    
    # Add grids
    [ax.grid(color='lightgrey', linewidth=0.5, zorder=0) for ax in axes]
    
    # Loop through the axes and parameters
    for ax, param, left, right in zip(axes, p.keys(), lefts, rights):
        
        # Plot the data
        ax.scatter(x=df[param], y=df['RMSE'],
                   facecolor='none',
                   edgecolors='black',
                   linewidth=0.5,
                   zorder=2)
        
        # Set the X-Axis
        ax.set_xlim(left=left, right=right)
        ax.set_xlabel(param, **font)
    
    # Set the Y-Axis
    [ax.set_ylim(bottom=0, top=2) for ax in [ax1, ax5]]
    [ax.set_ylabel('RMSE (m)', **font) for ax in [ax1, ax5]]
        
    # Save and close the figure
    title = f'BGB{profile} Parameters v RMSE'
    save_and_close(fig, title, profile, save)


def print_ga_best(hof, model, profile):
    """
    Print a neatly formatted Table of the best parameterizations
    from the genetic algorithm to a CSV
    
    hof: Halloffame object from the genetic algorithm
    model: Keras NN model
    profile: Int with the profile number
    """
    
    # Set empty lists
    sim, m, cb, facas, facsk, lsgrad, bedfric, wetslp, vegdensity, res =\
        [], [], [], [], [], [], [], [], [], []
    
    # Loop through the hof and populate the lists
    for ii, p in enumerate(hof):
        sim.append(ii)
        m.append(p[0])
        cb.append(p[1])
        facas.append(p[2])
        facsk.append(p[3])
        lsgrad.append(p[4])
        bedfric.append(p[5])
        wetslp.append(p[6])
        vegdensity.append(p[7])
        
        p = np.expand_dims(p, axis=0)
        res.append(model.predict(p)[0][0])
        
    # Convert the lists into a DataFrame
    df = pd.DataFrame.from_dict({
        'Sim': sim,
        'm': m,
        'Cb': cb,
        'facAs': facas,
        'facSk': facsk,
        'lsgrad': lsgrad,
        'bedfric': bedfric,
        'wetslp': wetslp,
        'vegdensity': vegdensity,
        'Expected RMSE': res})
    
    # Save the DataFrame to a .csv file
    if profile == None:
        fname = os.path.join('..', 'BGB15 and BGB22 NN', 'BGB15 and BGB22 Best ML Parameters.csv')
        pass
    else:
        fname = os.path.join('..', f'BGB{profile} NN', f'BGB{profile} Best ML Parameters.csv')
    df.to_csv(fname, index=False)
    print(f'File Saved: {fname}')
    

def rmse_kde(df, cut, p, save=False):
    """
    Make a KDE plot of the RMSE values below the
    cut value. Include a count of the total number
    of samples below the cut versus the overall total
    
    df: DataFrame with RMSE values
    cut: Value to cut the values at
    p: Int with the profile number
    save: Bool to display (False) or save (True) the figure
    """
    
    # Get stats
    total, n = len(df['RMSE']), len(df['RMSE'].loc[df['RMSE'] <= cut])
    min_val = np.around(np.nanmin(df['RMSE']), decimals=3)
    
    # Setup the figure
    fig, ax = plt.subplots(figsize=(inches, inches), dpi=dpi)
    
    # Add a grid
    ax.grid(color='lightgrey', linewidth=0.5, zorder=0)
    
    # Plot the KDE plot
    sns.kdeplot(df['RMSE'].loc[df['RMSE'] <= cut],
                color='black',
                shade=True,
                ax=ax,
                zorder=2)
    
    # Add text
    ax.text(x=1.0, y=3.45, s=f'n = {n} (of {total})\nMin = {min_val}', **font)
    
    # Set the X-Axis
    ax.set_xlim(left=0, right=2)
    ax.set_xlabel('RMSE$_{Windsurf}$ (m)', **font)
    
    # Set the Y-Axis
    ax.set_ylim(bottom=0, top=4)
    ax.set_ylabel('Density', **font)
    
    # Save and close the figure
    title = 'Windsurf RMSE Values'
    save_and_close(fig, title, p, save)


def save_and_close(fig, title, p, save):
    """
    Give the figure a tight and transparent background
    then close it

    fig: Figure object
    title: String with the title of the figure
    p: Int with the profile number
    save: Save the figure (True) of display it (False)
    """
    
    # Modify the figure directory to include the profile number
    if p == None:
        FIGURE_DIR = os.path.join('..', 'BGB15 and BGB22 NN')
    else:
        FIGURE_DIR = os.path.join('..', f'BGB{p} NN')

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
Functions to setup and run the machine learning algorithms
"""


def checkBounds(min, max):
    """
    Check that predicted parameter values from the
    genetic algorithm are within bounds before running
    the evaluation function on them. If a parameter is
    outside of the bounds, it is reset to a random number
    within the bounds
    
    Change the bounds in the set_bounds() function
    
    min: List of lower bounds
    max: List of upper bounds
    """
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    if i == 4:
                        new_val = np.nanmean([min[i], max[i]])
                    else:
                        difference = max[i] - min[i]
                        new_val = min[i] + (difference + random.random())
                        if new_val > max[i]:
                            new_val = max[i]
                        elif new_val < min[i]:
                            new_val = min[i]
                    if child[i] > max[i]:
                        child[i] = new_val
                    elif child[i] < min[i]:
                        child[i] = new_val
            return offspring
        return wrapper
    return decorator


def run_ga(model, bounds, n_hof, pop_size, cross_prob, mut_prob, num_gen):
    """
    Run the genetic algorithm on the trained neural network
    to predict optimal parameter values
    
    model: Trained neural network
    bounds: Dict with parameter names and upper/lower bounds
    n_hof: Int with the number of best predictions to store
    pop_size: Max number of individuals in each generation
    cross_prob: Float with the crossover probability
    mu_prob: Float with the mutation probability
    num_gen: Number of generations to run through the algorithm
    """
    
    # Set low and high boundaries
    low = [bounds[ii][0] for ii in bounds.keys()]
    hi = [bounds[ii][1] for ii in bounds.keys()]
    
    # Setup the individuals. The goal is to minimize the RMSE score, so
    # set the weight equal to -1. The individual will use a list of genes
    creator.create('FitnessMax', base.Fitness, weights=(-1.0, ))
    creator.create('Individual', list, fitness=creator.FitnessMax)
    
    # Register the individuals and population
    toolbox = base.Toolbox()
    toolbox.register('attr_m', random.uniform, bounds['m'][0], bounds['m'][1])
    toolbox.register('attr_cb', random.uniform, bounds['Cb'][0], bounds['Cb'][1])
    toolbox.register('attr_facas', random.uniform, bounds['facAs'][0], bounds['facAs'][1])
    toolbox.register('attr_facsk', random.uniform, bounds['facSk'][0], bounds['facSk'][1])
    toolbox.register('attr_lsgrad', random.uniform, bounds['lsgrad'][0], bounds['lsgrad'][1])
    toolbox.register('attr_bedfric', random.uniform, bounds['bedfric'][0], bounds['bedfric'][1])
    toolbox.register('attr_wetslp', random.uniform, bounds['wetslp'][0], bounds['wetslp'][1])
    toolbox.register('attr_vegdensity', random.uniform, bounds['vegDensity'][0], bounds['vegDensity'][1])
    toolbox.register('individual', tools.initCycle, creator.Individual,
                      (toolbox.attr_m, toolbox.attr_cb, toolbox.attr_facas, toolbox.attr_facsk, toolbox.attr_lsgrad,
                      toolbox.attr_bedfric, toolbox.attr_wetslp, toolbox.attr_vegdensity),
                      n=1)
    toolbox.register('population', tools.initRepeat, list, toolbox.individual)
    
    # Setup the mating strategy
    toolbox.register('mate', tools.cxBlend, alpha=0.5)
    toolbox.decorate("mate", checkBounds(low, hi))
    
    # Setup the mutation strategy
    toolbox.register('mutate', tools.mutGaussian, mu=0, sigma=1, indpb=0.2)
    toolbox.decorate("mutate", checkBounds(low, hi))
    
    # Setup parameter selection and evaluation
    toolbox.register('select', tools.selTournament, tournsize=3)
    toolbox.register('evaluate', windsurf_func, model=model)
    
    # Register the individuals and population
    hof = tools.HallOfFame(n_hof)

    # Register stats calculations
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Build a population
    population = toolbox.population(n=pop_size)

    # Run the algorithm
    pop, log = algorithms.eaSimple(population,
                                    toolbox,
                                    cxpb=cross_prob,
                                    mutpb=mut_prob,
                                    ngen=num_gen,
                                    verbose=True,
                                    stats=stats,
                                    halloffame=hof)
    
    return log, hof


def train_nn(df, bounds, rmse_cut, drop_rate, n_epoch, n_batch):
    """
    Train a neural network.
    
    df: DataFrame with values to train on and validate with
    bounds: Dict with parameter names and upper/lower bounds
    rmse_cut: Numeric value with the upper limit of RMSE values to train on
    drop_rate: Float with the dropout rate for the neural network
    n_epoch: Int with the number of epochs to train the network
    n_batch: Int with the batch size to train the network on 
    """
    
    # Remove runs with an RMSE greater than the cut value (set above)
    df = df.loc[df['RMSE'] <= rmse_cut]
    
    # Split the DataFrame into an X (parameters) and
    # y (RMSE) array
    X = df[bounds.keys()].to_numpy()
    y = df['RMSE'].to_numpy()
    
    # Split into training, testing and validation data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.25, random_state=42)
    
    # Scale the inputs
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_val = sc.fit_transform(X_val)
    X_test = sc.fit_transform(X_test)
    
    # Make a seqential model
    model = keras.Sequential([
        layers.Dense(8, input_shape=(8, ), activation='relu'),
        layers.Dropout(drop_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(drop_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(drop_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(drop_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(drop_rate),
        layers.Dense(128, activation='relu'),
        layers.Dropout(drop_rate),
        layers.Dense(1)
    ])

    # compile the keras model
    model.compile(loss='mean_squared_error', optimizer='Adam')
    
    # Fit the data
    history = model.fit(X_train,
                        y_train,
                        epochs=n_epoch,
                        batch_size=n_batch,
                        validation_data=(X_val, y_val))
    
    return model, history, X, y, X_train, X_test, X_val, y_train, y_test, y_val


def windsurf_func(individual, model):
    """
    Use the neural network to predict a Windsurf
    RMSE score using a set of parameters
    """

    # Predict an RMSE value
    params = np.array(individual)
    params = np.expand_dims(params, axis=0)
    ws_score = model.predict(params)

    # If the ws_score is a Nan then the model didn't complete,
    # this indicates the parameterization is particularly wrong
    # so set the score value to be extremely high
    if np.isnan(ws_score):
        ws_score = 1000.0

    return (ws_score,)


"""
Main program function
"""


def main():
    """
    Run the calibration
    """
    
    # Start a timer
    start_time = time.time()
    
    # Set the profile to analyze
    profile = 22

    # Set values for the Neural Network
    n_epoch = 200   # Should be good at 100
    n_batch = 100
    drop_rate = 0.10
    rmse_cut = 2
    
    # Set parameters for the genetic algorithm
    pop_size = 100
    num_gen = 100
    n_hof = 24
    cross_prob = 0.4
    mut_prob = 0.2
    
    # Turn sections on and off to make it quicker to test
    run_nn_section = True
    run_ga_section = True

    # Set the parameters tested
    bounds = set_bounds(profile, eps=0.05)

    """
    1. Compile all of the parameters and RMSE values from every
        BGB22 simulation into a DataFrame+
    """

    # Make the DataFrame and pull out the parameter names
    if profile == None:
        df15 = make_params(15)
        df22 = make_params(22)
        
        min_rmse_bgb15 = np.nanmin(df15['RMSE'])
        min_rmse_bgb22 = np.nanmin(df22['RMSE'])
        min_rmse = np.nanmean([min_rmse_bgb15, min_rmse_bgb22])
        
        df = pd.concat([df15, df22])
    else:
        df = make_params(profile)

    # Plot all the parameters versus RMSE
    params_v_rmse(df, bounds, profile, save=True)

    # Plot all the parameters against each other colored by RMSE
    parameter_pairplot(df, bounds, profile, save=True)
    
    """
    2. Train a neural network on the DataFrame
    """
    
    if run_nn_section:
    
        # Make a histogram of RMSE values
        rmse_kde(df, rmse_cut, profile, save=True)
        
        # Train the neural network
        model, history, X, y, X_train, X_test, X_val,\
            y_train, y_test, y_val =\
            train_nn(df, bounds, rmse_cut, drop_rate, n_epoch, n_batch)
        
        # Plot the loss from the training
        nn_loss_plot(history, profile, save=True)
        
        
        # Make a figure showing some test predictions
        nn_pred_obs_scatter(X_test, y_test, X_train, y_train, X_val, y_val, model, profile, save=True)
    
    """
    3. Find optimal parameter values by running a genetic
        algorithm through the neural network
    """
    
    if run_ga_section:
    
        # Run the algorithm
        log, hof = run_ga(model, bounds, n_hof, pop_size, cross_prob, mut_prob, num_gen)
        
        # Plot the loss from the genetic algorithm
        if profile == None:
            ga_loss_plot(log, min_val=min_rmse, profile=profile, save=True)
        else:
            ga_loss_plot(log, min_val=np.nanmin(y), profile=profile, save=True)
        
        # Print out the best parameterizations
        print_ga_best(hof, model, profile)
        
    # Print the elapsed time to the console
    end_time = time.time()
    print(f'--- {end_time - start_time} seconds ---')


if __name__ == '__main__':
    main()
