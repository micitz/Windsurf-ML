"""
Make Figures and perform analysis for Chapter 3: Modelling
Dune Erosion and Recovery on a Developed Barrier Island

Michael Itzkin, 11/10/2020
"""

from Functions import data_functions as dfuncs
from Functions import plot_functions as pfuncs
from Functions import text_functions as tfuncs


def main():
    """
    Main Program Function
    """

    """
    Load all the data needed for the analysis
    """

    # # Load the environmental data
    df = dfuncs.load_hindcast_data()

    # Load the profiles into DataFrames
    # bgb15 = dfuncs.load_overlays(p=15)
    # bgb22 = dfuncs.load_overlays(p=22)

    # Load field data
    # fm0, fmf = dfuncs.load_field_data(p=22, t0=2016, tf=2017)
    # ei0, eif = dfuncs.load_field_data(p=15, t0=2016, tf=2017)

    # Load field morphometrics
    # field_morpho = dfuncs.load_field_morpho()

    # Load model data
    # fenced_hindcast = dfuncs.load_model_data(experiment='Fenced Profile Hindcast', final=-1)
    # fm_hindcast = dfuncs.load_model_data(experiment='Fort Macon Hindcast', final=-1)

    # Load morphometrics from the model runs
    # fm_morpho = dfuncs.load_model_morpho(experiment='Fort Macon Hindcast', fence=None)
    # ei_morpho = dfuncs.load_model_morpho(experiment='Fenced Profile Hindcast', fence=-25)

    """
    Make the main figures
    
    These are the figures for the paper
    """

    # Plot the profile overlays
    # pfuncs.profile_overlays(fm=bgb22, ei=bgb15, save=True)

    # Make a barplot showing the profiles represented in
    # each category of dune growth and management level
    # pfuncs.dune_growth_barplots(save=True)
    # pfuncs.dune_and_beach_growth(save=True)

    # # Plot out the hindcast time series and make a text file with
    # # summary statistics about the environmental forcings. Also
    # # make a wind and wave rose
    # pfuncs.make_time_series_figure(df, save=True)
    pfuncs.wind_wave_rose(df, type='wave', save=True)
    pfuncs.wind_wave_rose(df, type='wind', save=True)
    # tfuncs.environmental_data_summary(df=df)
    #
    # # Plot the model final and initial profile overlain
    # # on the field initial and final profile
    # pfuncs.model_v_field_results(experiment='Fort Macon Hindcast', f0=fm0, ff=fmf, mm=fm_hindcast, save=True)
    # pfuncs.model_v_field_results(experiment='Emerald Isle Hindcast', f0=ei0, ff=eif, mm=fenced_hindcast, save=True)
    #
    # # Plot temporal changes to dune and beach elevation for
    # # Fort Macon and Atlantic Beach. Do a statistical test
    # # to see if differences are significant
    # pfuncs.fm_ab_v_time(fm=fm_morpho, ab=ei_morpho, save=False)

    """
    Secondary figures
    
    These might not be used in the paper but will
    be used in presentations and other places
    """

    # # Plot the fenced versus natural dune elevation
    # # for the Emerald Isle hindcast
    # pfuncs.fenced_v_natural_dune_height(df=ei_morpho, save=True)
    #
    # # Recreate figure 5 from Cohn et al. (2019) showing the
    # # change in shoreline, dune volume, and beach volume
    # # over the hindcast period.
    # pfuncs.cohn_figure_five(df=field_morpho, save=True)


if __name__ == '__main__':
    main()