"""
Run Windsurf using a genetic algorithm in order to tune its parameters
to best produce a hindcast of a desired field profile.

FUTURE GOAL: Tune via a hindcast of multiple profiles

Michael Itzkin, 2/2/2021
"""

from Windsurf import Windsurf

from deap import base, creator, tools, algorithms

import multiprocessing
import pandas as pd
import numpy as np
import random
import sys


def windsurf_func(individual):
    """
    Run a Windsurf instance and evaluate its output
    """

    test = True
    if test:
        ws_score_full, ws_score_dune, ws_score_beach = random.random(), random.random(), random.random()
        print(individual)
    else:

        # Check the system
        if sys.platform == 'win32':
            local = True
        elif sys.platform == 'linux2':
            local = False

        # Create a new Windsurf instance
        ws = Windsurf(profile=15, local=local, t0=2016, tf=2017, defaults=True)

        # Put the individuals into a dict to
        # pass onto the Windsurf constructor
        params_dict = {
            'm': individual[0],
            'Cb': individual[1],
            'facAsHi': individual[2],
            'facAsLo': individual[2],
            'facSkHi': individual[3],
            'facSkLo': individual[3],
            'lsgrad': individual[4],
            'bedfric': individual[5],
            'wetslope': individual[6],
            'vegDensity': individual[7],
        }
        ws.set_params(params_dict, save=True, newfolder=True)

        # Run the model and calculate the RMSE
        ws.run_windsurf()
        ws_score_full, ws_score_dune, ws_score_beach = ws.evaluate_sim(dec=None)

        # If the ws_score is a Nan then the model didn't complete,
        # this indicates the parameterization is particularly wrong
        # so set the score value to be extremely high
        # if np.isnan(ws_score):
        #     ws_score = 1000.0

    return (ws_score_full, ws_score_dune, ws_score_beach,)


"""
Run the algorithm
"""


def checkBounds(min, max):
    def decorator(func):
        def wrapper(*args, **kargs):
            offspring = func(*args, **kargs)
            for child in offspring:
                for i in range(len(child)):
                    difference = max[i] - min[i]
                    new_val = min[i] + (difference * random.random())
                    if child[i] > max[i]:
                        child[i] = new_val
                    elif child[i] < min[i]:
                        child[i] = new_val
            return offspring
        return wrapper
    return decorator


def main(pop_size, num_gen):
    """
    Main loop for running and tuning the model
    """

    # Register the individuals and population
    pool = multiprocessing.Pool(processes=pop_size + 1)
    hof = tools.HallOfFame(1)
    toolbox.register('map', pool.map)

    # Register stats calculations
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Build a population
    population = toolbox.population(n=pop_size)

    # Run the algorithm
    pop, log = algorithms.eaSimple(population, toolbox, cxpb=0.4, mutpb=0.2, ngen=num_gen, verbose=True, halloffame=hof)

    pool.close()


"""
Setup the algorithm
"""

# Load in the values from the neural network and set the
# percentage to mulitply the values by for seeding the
# first generation
pct = 0.05
nn_fname = 'BGB15 Best ML Parameters.csv'
nn_df = pd.read_csv(nn_fname, delimiter=',', header=0)
nn_means = nn_df.mean(axis=0)
means_df = pd.DataFrame({'Parameter': nn_means.index, 'Mean': nn_means.values})
means_df.set_index('Parameter', inplace=True)
means_df['Lo'] = means_df['Mean'] * (1 - pct)
means_df['Hi'] = means_df['Mean'] * (1 + pct)

# Set parameters for the algorithm
pop_size = 10
num_gen = 10
n_genes = 8

# Set upper and lower bounds for the parameters. This is
# in the same order the parameters are registered below
low = [0.00, 0.00, 0.00, 0.00, -0.10, 0.00, 0.00, 0.00]
hi = [5.00, 3.00, 1.00, 1.00, 0.10, 0.50, 1.00, 10.00]

# Setup the individuals. The goal is to minimize the RMSE score, so
# set the weight equal to -1. The individual will use a list of genes
creator.create('FitnessMulti', base.Fitness, weights=(-1.0, -1.0, -1.0))
creator.create('Individual', list, fitness=creator.FitnessMulti)

# Register the individuals and population
toolbox = base.Toolbox()
toolbox.register('attr_m', random.uniform, means_df['Lo'].loc['m'], means_df['Hi'].loc['m'])
toolbox.register('attr_cb', random.uniform, means_df['Lo'].loc['Cb'], means_df['Hi'].loc['Cb'])
toolbox.register('attr_facas', random.uniform, means_df['Lo'].loc['facAs'], means_df['Hi'].loc['facAs'])
toolbox.register('attr_facsk', random.uniform, means_df['Lo'].loc['facSk'], means_df['Hi'].loc['facSk'])
toolbox.register('attr_lsgrad', random.uniform, means_df['Lo'].loc['lsgrad'], means_df['Hi'].loc['lsgrad'])
toolbox.register('attr_bedfric', random.uniform, means_df['Lo'].loc['bedfric'], means_df['Hi'].loc['bedfric'])
toolbox.register('attr_wetslp', random.uniform, means_df['Lo'].loc['wetslp'], means_df['Hi'].loc['wetslp'])
toolbox.register('attr_vegdensity', random.uniform, means_df['Lo'].loc['vegdensity'], means_df['Hi'].loc['vegdensity'])
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
toolbox.register('evaluate', windsurf_func)


if __name__ == '__main__':
    main(pop_size, num_gen)
