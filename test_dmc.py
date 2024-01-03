import pytest 
import numpy as np
from polaron import Polaron
from dmc import run_thermalization_steps, run_diagrammatic_montecarlo
from unittest.mock import patch
from test_polaron import polaron, polaron_order_two, phonon

@pytest.mark.parametrize("polarons", ["polaron", "polaron_order_two"])
def test_run_thermalization_steps_conditions(polarons, request):
    """This test checks that run_thermalization_steps 
    calls the correct updates depending on the conditional
    statement that is satisfied. 

    When a zero order polaron is supplied in the first step 
    we expect to have no calls to numpy.random.randint

    Instead for a polaron with order > 0 the call has to 
    be done as specified in the conditional statement
    
    To check the actual behaviour and the number of calls 
    numpy.random.randint is mocked using patch from unittest.mock
    
    GIVEN: a polaron object
    WHAT: apply run_thermalization_steps with one step
    THEN: mock_randint should not be called if initial order is 0
        and called once if initial order > 0
    """

    polaron = request.getfixturevalue(polarons)
    initial_order = polaron.diagram['order']

    # set seed to assert deterministic outcome from np.random.uniform
    np.random.seed(1)
    steps = 1 

    # mock the calls to int random number to choose update
    with patch("numpy.random.randint") as mock_randint:
        mock_randint.return_value = 0
        thermalized_polaron = run_thermalization_steps(polaron, steps)
    
    if initial_order == 0:
        mock_randint.assert_not_called()
        assert isinstance(thermalized_polaron, Polaron)
        assert thermalized_polaron.diagram['order'] >= 0
    elif initial_order > 0:
        mock_randint.assert_called_once()
        assert isinstance(thermalized_polaron, Polaron)
        assert thermalized_polaron.diagram['order'] >= 0

def test_run_thermalization_steps_nsteps(polaron):
    """This test checks that run_thermalization_steps make exactly
    the expected number of MonteCarlo steps from an initial configuration
    with a zero order diagram. 

    When a zero order polaron is supplied in the first step
    there is always a call for eval_add_internal which is the only
    valid update. 
    Mocking the update polaron.Polaron.eval_add_internal
    it is possible to check if the function makes exactly
    the expected number of steps since the update is called exactly
    one time at each step
    
    
    GIVEN: a zero order polaron object
    WHAT: apply run_thermalization_steps with nsteps_burn MonteCarlo steps
    THEN: check that the mock function for eval_add_update
      is called exactly the expected number of times (i.e. number of steps)
    """
    steps = 10 

    with patch("polaron.Polaron.eval_add_internal") as mock_eval_add_internal:
        _ = run_thermalization_steps(polaron, steps)
    
    assert mock_eval_add_internal.call_count == steps

@pytest.mark.parametrize("polarons", ["polaron", "polaron_order_two"])
def test_run_diagrammatic_montecarlo_conditions(polarons, request):
    """This test checks that run_diagrammatic_montecarlo 
    calls the correct updates depending on the conditional
    statement that is satisfied. 

    When a zero order polaron is supplied in the first step 
    we expect to have no calls to numpy.random.randint

    Instead for a polaron with order > 0 the call has to 
    be done as specified in the conditional statement
    
    To check the actual behaviour and the number of calls 
    numpy.random.randint is mocked using patch from unittest.mock
    
    GIVEN: a polaron object
    WHAT: apply run_diagrammatic_montecarlo with one step
    THEN: mock_randint should not be called if initial order is 0
        and called once if initial order > 0
    """

    polaron = request.getfixturevalue(polarons)
    initial_order = polaron.diagram['order']

    # set seed to assert deterministic outcome from np.random.uniform
    np.random.seed(1)
    steps = 1 

    # mock the calls to int random number to choose update
    with patch("numpy.random.randint") as mock_randint:
        mock_randint.return_value = 0
        _,_ = run_diagrammatic_montecarlo(polaron, steps)
    
    if initial_order == 0:
        mock_randint.assert_not_called()
    elif initial_order > 0:
        mock_randint.assert_called_once()

@pytest.mark.parametrize("polarons", ["polaron", "polaron_order_two"])
def test_run_diagrammatic_montecarlo_nsteps(polarons, request):
    """This test checks that run_diagrammatic_montecarlo make
    exactly the expected number of MonteCarlo steps both from an 
    initial configuration with a zero order diagram and with a 
    non zero order diagram . 
 
    When a zero order polaron is supplied in the first step
    there is always a call for eval_add_internal which is the only
    valid update, instead if the supplied polaron has order > 0 there
    is the possibility to pick either add or remove.

    Mocking both numpy.random.randint to call always the add 
    update and the function polaron.Polaron.eval_add_internal
    it is possible to check if the function makes exactly
    the expected number of steps since the update is called exactly
    one time at each step
    
    
    GIVEN: a zero order polaron object
    WHAT: apply run_diagrammatic_montecarlo with steps
    THEN: check that the mock function for eval_add_update
      is called exactly the expected number of times (i.e. number of steps)
    """
    polaron = request.getfixturevalue(polarons)
    steps = 10 

    with patch("numpy.random.randint") as mock_randint,  \
        patch("polaron.Polaron.eval_add_internal") as mock_eval_add_internal:
        mock_randint.return_value = 0
        _,_ = run_diagrammatic_montecarlo(polaron, steps)

    assert mock_eval_add_internal.call_count == steps


