"""
READ: 
-simulation parameters from file

PERFORM :
-MonteCarlo simulation for polaronic diagrams

STORE :
-sequences of diagrams 

PLOT :
-distribution of orders for sampled diagrams
-distribution of number of phonons presents in each diagram
"""

import numpy as np
import plot
from dmc import run_diagrammatic_montecarlo, run_thermalization_steps
from polaron import Polaron
import config_parser
from sys import argv

if __name__ == "__main__":

    config = config_parser.Config(argv[1])

    #get settings parameter for the simulation
    settings = config.get_settings()
    
    #check parameters are positive
    config_parser.check_positive_parameters(settings)

    #get path to store plot 
    path_plot = config.get_path_plot()

    #get path to store data
    path_data = config.get_path_data()
    
    #ensure storage directories exist
    config_parser.ensure_storage_directories_exist(path_plot,path_data)

    polaron=Polaron(settings['OMEGA'], settings['G'], 
                    settings['TIME'])
    
    #setting seed of the random sequence to get reproducibility
    np.random.seed(1)

    if settings['NSTEPS_BURN'] > 0:
        print("Starting thermalization of Markov chain" + '\n'
              + "----------------------------------------")
        polaron = run_thermalization_steps(polaron, settings['NSTEPS_BURN'])
        print("Thermalization steps for Markov chain ended :")
        print(f"""Starting features in the Feynman diagram of the polaron:
        - Diagram order : {polaron.diagram['order']}
        - Lifetime : {polaron.diagram['time']}""" + '\n' +
        "----------------------------------------")

    print(f'Starting MonteCarlo simulation of {settings["""NSTEPS"""]} steps' + '\n' 
            + "----------------------------------------")
    order_sequence, energy_sequence = run_diagrammatic_montecarlo(polaron, settings['NSTEPS'])
    print('Simulation ended!')
    plot.plot_montecarlo(order_sequence, energy_sequence, 
                         settings, path_plot, path_data)
