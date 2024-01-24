import numpy as np
from polaron import Polaron
from dmc import run_thermalization_steps, run_diagrammatic_montecarlo
from test_polaron import polaron

def test_run_thermalization_steps(polaron):
    """This test checks that run_thermalization_steps 
    returns a valid Polaron object and the lists of 
    diagram order and energy empty since these steps
    are not used for the stastical measurements. 
    
    GIVEN: a polaron object
    WHAT: apply run_thermalization_steps with a fixed number of steps
    THEN: return a valid Polaron with non negative diagram order
    """
    np.random.seed(1)
    steps = 10 

    thermalized_polaron = run_thermalization_steps(polaron, steps)

    assert isinstance(thermalized_polaron, Polaron)
    assert thermalized_polaron.diagram['order'] >= 0

def test_run_diagrammatic_montecarlo(polaron):
    """This test checks that run_diagrammatic_montecarlo 
    calls returns the lists of diagram order and energy
    with length equals to the number of steps. 
    
    GIVEN: a polaron object
    WHAT: apply run_diagrammatic_montecarlo with some steps
    THEN: return the lists of diagram order and energy with a length
        equals to the number of steps
    """
    np.random.seed(1)
    steps = 10 

    order_sequence, energy_sequence = run_diagrammatic_montecarlo(polaron, steps)
    
    assert len(order_sequence) == steps
    assert len(energy_sequence) == steps




