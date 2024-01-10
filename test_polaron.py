import pytest
import numpy as np
from unittest.mock import patch
from polaron import Polaron
from math import isclose

@pytest.fixture
def polaron():
    """This method return a fixed polaron object
    that can be used consistently during the tests
    """
    return Polaron(omega=1.0, g=0.5, time=10.0)

@pytest.fixture
def phonon():
    """This method return a np.array of two times
    that can be used consistently as a phonon during
    the tests
    """
    return np.array([0.2,0.5])

def test_polaron_initialization(polaron):
    """This test checks that the attribute of the Polaron
    class are initialized correctly
    
    GIVEN: polaron object
    WHAT: check the attributes are initialized accordingly 
        to the given parameters
    THEN: get a valid polaron object
    """
    assert isclose(polaron.diagram['omega'], 1.0)
    assert isclose(polaron.diagram['g'], 0.5)
    assert isclose(polaron.diagram['time'], 10.0)
    assert polaron.diagram['order'] == 0
    assert isclose(polaron.diagram['total_energy'], 0.0)
    assert polaron.phonon_list == []
    assert polaron.order_sequence == []
    assert polaron.energy_sequence == []

def test_metropolis(polaron):
    """This test checks that Metropolis returns the correct values
    accordingly to the given acceptance probabilities
    
    GIVEN: a ratio between acceptance probabilities
    WHAT: apply metropolis function
    THEN: get 1.0 if the ratio > 1.0 or the value if it is < 1.0
    """
    assert isclose(polaron.metropolis(0.7), 0.7)
    assert isclose(polaron.metropolis(1.2), 1.0)

def test_add_phonon_scaling(polaron):
    """This test checks that a correct factor due to imaginary
    scaled times is put in front of the ratio between the weight
    of the diagrams

    GIVEN: a polaron object with specific parameters
    WHAT: apply add_phonon_scaling factor
    THEN: the scaling factor has to be equal to the one
        evaluated manually with the analytic formula
    """
    scaling_factor = polaron.add_phonon_scaling()
    assert isclose(scaling_factor, (0.5 * 10.0) ** 2)

def test_weigth_ratio_add(polaron, phonon):
    """This test checks whether the ratio between a diagram
    with one more phonon and the current one has the expected value
    
    GIVEN: a polaron object with specific parameters
        and a phonon with specific times
    WHAT: apply the weight_ratio_add to evaluate the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula
    """
    ratio = polaron.weigth_ratio_add(phonon)
    expected_ratio = (0.5 * 10.0) ** 2 * np.exp(-10.0 * 1.0 * (0.5 - 0.2))
    assert isclose(ratio,expected_ratio) 
    
def test_proposal_add_ratio_zero_order(polaron,phonon):
    """This test checks whether the ratio between the proposal
    probabilities of the reverse and direct process for the 
    add_phonon update has the expected value
    
    GIVEN: a polaron object with specific parameters
        and a phonon with specific times
    WHAT: apply the function proposal_add_ratio to evaluate 
        the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula

    Notes:
        This test refers to the fixture polaron with 
        a zero order diagram
    """
    ratio = polaron.proposal_add_ratio(phonon)
    assert polaron.diagram['order'] == 0
    expected_ratio = 0.5 * (1 - 0.2 )/(0 + 1)
    assert isclose(ratio,expected_ratio)

def test_add_internal_zero_order(polaron, phonon):
    """This test checks that the add_internal method
    adds a phonon to the phonon_list and updates the order
    
    GIVEN: a polaron object with specific parameters
        and a phonon with specific times
    WHAT: apply the function add_internal to add a phonon
        to the phonon list and update the order 
    THEN: the phonon_list should have a phonon and the order
        of the diagram should be increased by 2

    Notes:
        This test refers to the fixture polaron with 
        a zero order diagram
    """
    assert polaron.phonon_list == []
    polaron.add_internal(phonon)
    expected_phonon_list=[phonon]
    assert all(np.allclose(actual,expected) for actual,expected in 
                              zip(polaron.phonon_list,expected_phonon_list))
    assert polaron.diagram['order'] == 2

@pytest.fixture
def polaron_order_two(phonon):
    """This method return a fixed polaron object
    with a phonon that can be used consistently
    during the tests
    """
    polaron = Polaron(omega=1.0, g=0.5, time=10.0)
    polaron.add_internal(phonon)
    return polaron

@pytest.fixture
def another_phonon():
    """This method return a np.array of two times
    that can be used consistently as a phonon during
    the tests
    """
    return np.array([0.6,0.7])

@pytest.mark.parametrize("polarons", ["polaron", "polaron_order_two"])
def test_proposal_add_ratio(polarons, another_phonon, request):
    """This test checks whether the ratio between the proposal
    probabilities of the reverse and direct process for the 
    add_phonon update has the expected value
    
    GIVEN: a polaron object with specific parameters
        and a phonon with specific times
    WHAT: apply the function proposal_add_ratio to evaluate 
        the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula
    """
    polaron = request.getfixturevalue(polarons)
    ratio = polaron.proposal_add_ratio(another_phonon)
    if polaron.diagram['order'] == 0:
        expected_ratio = 0.5 * (1 - 0.6)/(0 + 1)
    else:
        expected_ratio = (1 - 0.6)/(len(polaron.phonon_list) + 1)
    assert isclose(ratio,expected_ratio)

@pytest.mark.parametrize("polarons", ["polaron", "polaron_order_two"])
def test_add_internal(polarons, another_phonon, request):
    """This test checks that the add_internal method
    adds a phonon to the phonon_list and updates the order
    
    GIVEN: a polaron object with specific parameters
        and a phonon with specific times
    WHAT: apply the function add_internal to add a phonon
        to the phonon list and update the order 
    THEN: the phonon_list should have a phonon and the order
        of the diagram should be increased by 2
    """
    polaron = request.getfixturevalue(polarons)
    #store initial parameters
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_diagram_order = polaron.diagram['order']
    #store initial phonon list
    initial_phonon_list = polaron.phonon_list[:]

    #call to the function
    polaron.add_internal(another_phonon)

    #check diagram order and lenght of phonon list is updated correctly
    assert len(polaron.phonon_list) == initial_phonon_list_length + 1
    assert polaron.diagram['order'] == initial_diagram_order + 2
    #check lists are equal phonon by phonon
    expected_phonon_list=initial_phonon_list + [another_phonon]
    assert all(np.allclose(actual,expected) for actual,expected in 
                              zip(polaron.phonon_list, expected_phonon_list))


@pytest.mark.parametrize("parameters", [
    {'t_gen': 0.3, 't_rem': 0.5, 'expected_acceptance': 1.0},
    {'t_gen': 0.6, 't_rem': 0.8, 'expected_acceptance': 0.68, 'sampled_acceptance' : 0.5},
    {'t_gen': 0.6, 't_rem': 0.8, 'expected_acceptance': 0.68, 'sampled_acceptance' : 0.8},
])
def test_eval_add_internal(polaron, parameters):
    """This test checks the behaviour of the eval_add_internal method.
     
    It is responsible for evaluating the acceptance probability for the
    introduction of a phonon and eventually adding it to the current 
    diagram depending on the outcome of Metropolis-Hastings.

    The test involves patching the random number generation to control 
    the random values used in the evaluation. The method is expected 
    to add a phonon to the phonon_list and update the order of the diagram
    based on the Metropolis-Hastings acceptance criterion.

    Parameters:
        polaron: A Polaron instance with a specific configuration.
        parameters: A list of dictionaries with the patched random values
              and the expected acceptance
    """
    # Get the initial state
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_order = polaron.diagram['order']

    if isclose(parameters['expected_acceptance'],1.0):
        # Patch the random.uniform function to control its output
        with patch('numpy.random.uniform') as mock_uniform:
            mock_uniform.side_effect = [parameters['t_gen'], 
                            parameters['t_rem']]
            # Call the method under test
            polaron.eval_add_internal()

        assert len(polaron.phonon_list) == initial_phonon_list_length + 1
        assert polaron.diagram['order'] == initial_order + 2

    elif parameters['expected_acceptance'] < 1.0:
        with patch('numpy.random.uniform') as mock_uniform:
            mock_uniform.side_effect = [parameters['t_gen'], 
                            parameters['t_rem'], parameters['sampled_acceptance']]
            polaron.eval_add_internal()

        if parameters['sampled_acceptance'] < parameters['expected_acceptance']:
            assert len(polaron.phonon_list) == initial_phonon_list_length + 1
            assert polaron.diagram['order'] == initial_order + 2
        else:
            assert len(polaron.phonon_list) == initial_phonon_list_length
            assert polaron.diagram['order'] == initial_order

@pytest.fixture
def polaron_order_four(phonon, another_phonon):
    """This method return a fixed polaron object
    with two phonons that can be used consistently
    during the tests
    """
    polaron = Polaron(omega=1.0, g=0.5, time=10.0)
    polaron.add_internal(phonon)
    polaron.add_internal(another_phonon)
    return polaron

def test_weigth_ratio_remove(polaron_order_two, phonon):
    """This test checks whether the ratio between a diagram
    with one less phonon and the current one gives the expected value
    
    GIVEN: a polaron object of order two with
      specific parameters and a phonon with specific times
    WHAT: apply the weight_ratio_remove to evaluate the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula
    """
    ratio = polaron_order_two.weigth_ratio_remove(phonon)
    expected_ratio =  np.exp(10.0 * 1.0 * (0.5 - 0.2)) / ((0.5 * 10.0) ** 2)
    assert isclose(ratio,expected_ratio)

@pytest.mark.parametrize("polarons", ["polaron_order_four", 
                                      "polaron_order_two"])
def test_proposal_remove_ratio(polarons, phonon, request):
    """This test checks whether the ratio between the proposal
    probabilities of the reverse and direct process for the 
    remove_phonon update gives the expected value
    
    GIVEN: a polaron object with specific parameters
        and a phonon with specific times
    WHAT: apply the function proposal_remove_ratio to evaluate 
        the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula
    """
    polaron = request.getfixturevalue(polarons)
    ratio = polaron.proposal_remove_ratio(phonon)
    if polaron.diagram['order'] == 2:
        expected_ratio = 2 * 1/(1 - 0.2)
    else:
        expected_ratio = (len(polaron.phonon_list))/(1 - 0.2)
    assert isclose(ratio,expected_ratio)

@pytest.mark.parametrize("phonon_index", [0,1])
def test_choose_phonon(polaron_order_four, phonon_index):
    """This test checks whether the choose_phonon 
    function retrieves an index from the ones
    in phonon_list.

    This tests mocks the random sampling of integers 
    from np.random.randint and checks that during the test
    the function in called exactly one time as expected
    
    GIVEN: a polaron object with specific parameters
        and a phonon index
    WHAT: apply the function choose phonon to evaluate 
        the index
    THEN: the actual index has to be equal to the 
        expected index provided as a parameter
    """
    with patch('numpy.random.randint') as mock_randint:
       
        mock_randint.return_value = phonon_index
        actual_index = polaron_order_four.choose_phonon()

    mock_randint.assert_called_once_with(len(polaron_order_four.phonon_list))
    assert actual_index == phonon_index

@pytest.mark.parametrize("parameters", [
    {'polaron': "polaron_order_four", 'phonon_index': 0},
    {'polaron': "polaron_order_two", 'phonon_index' :0}])
def test_remove_internal(parameters, request):
    """This test checks that the remove_internal method
    removes a phonon from the phonon_list and updates the order
    
    GIVEN: a polaron object with specific parameters
        and a specific phonon_index 
    WHAT: apply the function remove_internal to remove a phonon
        to the phonon list and update the order 
    THEN: the phonon_list should have no more the phonon indexed
        by the parameter and the order of the diagram
        should be decreased by 2
    """
    polaron = request.getfixturevalue(parameters['polaron'])
    #store initial parameters
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_diagram_order = polaron.diagram['order']
    #get expected phonon list after removal
    expected_phonon_list = [phonon for i,phonon in 
                            enumerate(polaron.phonon_list) if i!=0]
    #call to the function, removing first phonon [0.2,0.5]
    polaron.remove_internal(parameters['phonon_index'])

    #check diagram order and lenght of phonon list is updated correctly
    assert len(polaron.phonon_list) == initial_phonon_list_length - 1
    assert polaron.diagram['order'] == initial_diagram_order - 2
    #check lists are equal phonon by phonon
    assert all(np.allclose(actual,expected) for actual,expected in 
                              zip(polaron.phonon_list, expected_phonon_list))

@pytest.mark.parametrize("parameters", [
    {'polaron': "polaron_order_two", 'phonon_index' : 0, 'expected_acceptance': 1.0},
    {'polaron': "polaron_order_four", 'phonon_index' : 1, 'expected_acceptance': 0.54, 'sampled_acceptance' : 0.3},
    {'polaron': "polaron_order_four", 'phonon_index' : 1, 'expected_acceptance': 0.54, 'sampled_acceptance' : 0.7},
])
def test_eval_remove_internal(request, parameters):
    """This test checks the behaviour of the eval_remove_internal method.
     
    It is responsible for evaluating the acceptance probability for the
    removal of a phonon and eventually removing it from the current 
    diagram depending on the outcome of Metropolis-Hastings.

    The test involves patching the random number generation to control 
    the random values used in the evaluation. The method is expected 
    to remove a phonon from the phonon_list and update the order of the diagram
    based on the Metropolis-Hastings acceptance criterion.

    Parameters:
        request: a fixture that can be used to access information
                from the provided function
        parameters: A list of dictionaries with the patched random values
              a specific polaron and the expected acceptance
    """
    # Get the initial state
    polaron=request.getfixturevalue(parameters['polaron'])
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_order = polaron.diagram['order']

    if isclose(parameters['expected_acceptance'],1.0):
        # Patch the random.uniform function to control its output
        with patch('numpy.random.randint') as mock_randint:
            mock_randint.return_value = parameters['phonon_index']  # Replace with the desired index
            # Call the method under test
            polaron.eval_remove_internal()

        assert len(polaron.phonon_list) == initial_phonon_list_length - 1
        assert polaron.diagram['order'] == initial_order - 2

    elif parameters['expected_acceptance'] < 1.0:
        #setting calls to mock functions for 
        with patch('numpy.random.randint') as mock_randint, \
         patch('numpy.random.uniform') as mock_uniform:

            mock_randint.return_value = parameters['phonon_index']
            mock_uniform.return_value = parameters['sampled_acceptance']

            polaron.eval_remove_internal()

        if parameters['sampled_acceptance'] < parameters['expected_acceptance']:
            assert len(polaron.phonon_list) == initial_phonon_list_length - 1
            assert polaron.diagram['order'] == initial_order - 2
        else:
            assert len(polaron.phonon_list) == initial_phonon_list_length
            assert polaron.diagram['order'] == initial_order

@pytest.mark.parametrize("polarons", 
            ["polaron", "polaron_order_two"])
def test_eval_diagram_energy(polarons, request):
        """Evaluate energy of the system at a MonteCarlo step using the formula of the 
        estimator
        """
        polaron = request.getfixturevalue(polarons)

        if polaron.diagram['order'] == 0:
            polaron.eval_diagram_energy()
            assert isclose(polaron.diagram['total_energy'],0.0)
        elif polaron.diagram['order'] > 0:
            assert polaron.diagram['order'] == 2
            phonons_interaction_time_sum = 0.5 - 0.2
            actual_energy = 1.0 * phonons_interaction_time_sum - 2 / 10.0
            #calls to the function eval_diagram_energy
            polaron.eval_diagram_energy()
            assert isclose(polaron.diagram['total_energy'],actual_energy)

def test_update_diagrams_info(polaron_order_two):
    """This test checks that the lists with diagrams order and energy
       of the polaron are updated correctly during the simulation.
       Both for diagram diagrams got after the thermalization

       GIVEN: a polaron object
       WHAT : apply set_starting_info function
       THEN : have two lists with the correct 
            corresponding values for order and energy
    """
    #simulating zero order diagram at first step
    polaron_order_two.order_sequence.append(0)
    polaron_order_two.energy_sequence.append(0.0)

    initial_order_list = polaron_order_two.order_sequence.copy()
    initial_energy_list = polaron_order_two.energy_sequence.copy()

    #assert actual lists are equal before updating
    assert initial_order_list == polaron_order_two.order_sequence
    assert initial_energy_list == polaron_order_two.energy_sequence

    #get order and energy of polaron_order_two
    order = polaron_order_two.diagram['order']
    polaron_order_two.eval_diagram_energy()
    energy = polaron_order_two.diagram['total_energy']

    polaron_order_two.update_diagrams_info()

    expected_order_list = initial_order_list + [order]
    expected_energy_list = initial_energy_list + [energy]

    assert polaron_order_two.order_sequence == expected_order_list  
    assert np.allclose(polaron_order_two.energy_sequence, expected_energy_list, rtol=1e-9)
