"""
Helper functions that perform the diagrammatic MonteCarlo simulation for the
Holstein polaron.
"""

import numpy as np
from polaron import Polaron

def run_thermalization_steps(polaron : Polaron, nsteps_burn : int) -> Polaron :
    """This function run some steps previous MonteCarlo steps 
    to thermalize the Markov chain
    
    Parameters:
     polaron : polaron object with the diagram and other features
     nsteps_burn : number of MonteCarlo steps employed to thermalize the chain

    Return:
        polaron: polaron object with an updated diagram and eventually 
        the list of phonons if the order is greater than 0
    ------------------------
    Notes:
    These steps are used to thermalize the Markov chain so the observables
    are not stored 
    """
    for _ in range(nsteps_burn):
        if polaron.diagram['order'] == 0:
            polaron.eval_add_internal()
        else:
            number = np.random.randint(2)
            if number == 0:
                polaron.eval_add_internal()
            else:
                polaron.eval_remove_internal()

    return polaron

def run_diagrammatic_montecarlo(polaron : Polaron, nsteps : int) :
    """This function run the steps of the MonteCarlo simulation
    
    Parameters:
     polaron : polaron object with the diagram and other features
     nsteps : number of MonteCarlo steps employed  in the simulation

    Return:
        tuple(list,list) that contains the sequences of orders and energies
        of each diagram sampled in the simulation
    """
    for _ in range(nsteps):
        if polaron.diagram['order'] == 0:
            polaron.eval_add_internal()
        else:
            number = np.random.randint(2)
            if number == 0:
                polaron.eval_add_internal()
            else:
                polaron.eval_remove_internal()

        polaron.eval_diagram_energy()
        polaron.update_diagrams_info()

    return (polaron.order_sequence, polaron.energy_sequence)
