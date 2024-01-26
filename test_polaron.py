import pytest
import numpy as np
from unittest.mock import patch
from polaron import Polaron
from math import isclose

@pytest.fixture
def polaron():
    """This method returns a fixed polaron object
    that can be used consistently during the tests

    Return:
        polaron object with zero order
    """
    return Polaron(omega=1.0, g=0.5, time=10.0)

@pytest.fixture
def phonon():
    """This method returns a np.array of two times
    that can be used consistently as a phonon during
    the tests

    Return:
        phonon with generation and removal times
    """
    return np.array([0.2,0.5])

def test_polaron_initialization(polaron):
    """This test checks that the attribute of the Polaron
    class are initialized correctly
    
    Parameters:
        polaron: fixture polaron

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
    
    Parameters:
        polaron: fixture polaron

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

    Parameters:
        polaron: fixture polaron

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
    
    Parameters:
        polaron: fixture polaron
        phonon: fixture phonon

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

    Parameters:
        polaron: fixture polaron
        phonon: fixture phonon
    
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
    expected_ratio = 0.5 * (1 - 0.2)/(0 + 1)
    assert isclose(ratio,expected_ratio)

def test_add_internal_zero_order(polaron, phonon):
    """This test checks that the add_internal method
    adds a phonon to the phonon_list and updates the order
    
    Parameters:
        polaron: fixture polaron
        phonon: fixture phonon

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

    Parameters:
        phonon: phonon fixture
    Return:
        polaron: polaron object with order two 
            and a phonon
    """
    polaron = Polaron(omega=1.0, g=0.5, time=10.0)
    polaron.add_internal(phonon)
    return polaron

@pytest.fixture
def another_phonon():
    """This method return a np.array of two times
    that can be used consistently as a phonon during
    the tests

    Return:
        phonon with generation and removal times
    """
    return np.array([0.6,0.7])

def test_proposal_add_ratio_zero_order(polaron, phonon):
    """This test checks whether the ratio between the proposal
    probabilities of the reverse and direct process for the 
    add_phonon update has the expected value for a polaron
    of order zero

    Parameters:
        polaron: fixture polaron
        phonon: fixture phonon
    
    GIVEN: a polaron object of order zero and a phonon 
        with specific times
    WHAT: apply the function proposal_add_ratio to evaluate 
        the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula
    """
    ratio = polaron.proposal_add_ratio(phonon)
    expected_ratio = 0.5 * (1 - 0.2)/(0 + 1)
    assert isclose(ratio,expected_ratio)

def test_proposal_add_ratio_order_not_zero(polaron_order_two, another_phonon):
    """This test checks whether the ratio between the proposal
    probabilities of the reverse and direct process for the 
    add_phonon update has the expected value for a polaron
    with order greater than zero

    Parameters:
        polaron_order_two: fixture polaron_order_two
        another_phonon: fixture another_phonon
    
    GIVEN: a polaron object with order greater than zero
        and a phonon with specific times
    WHAT: apply the function proposal_add_ratio to evaluate 
        the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula
    """
    ratio = polaron_order_two.proposal_add_ratio(another_phonon)
    expected_ratio = (1 - 0.6)/(len(polaron_order_two.phonon_list) + 1)
    assert isclose(ratio,expected_ratio)

@pytest.mark.parametrize("polarons", ["polaron", "polaron_order_two"])
def test_add_internal(polarons, another_phonon, request):
    """This test checks that the add_internal method
    adds a phonon to the phonon_list and updates the order
    The test is carried out both for a polaron of zero order
    and order greater than zero using pytest.mark.parametrize

    Parameters:
        polarons: lists of string with the name of the polaron
            parametrize at each execution of the test 
        another_phonon: fixture another_phonon
        request: pytest fixture used to access the value of a fixture
            having its name passed as a string
    
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

@pytest.fixture
def parameters_eval_add_internal():
    """This fixture returns a list of dictionaries where each contains the
    relevant structure and internal parameters used in the tests of
    eval_add_internal:

     1) time at which the phonon is generated
     2) time at which the phonon is removed
     3) the acceptance probability that the add_internal update is accepted.
        It is evaluated manually considering the two times of the phonon
        and a polaron of order zero
     4) the sampled acceptance probability from Metropolis-Hastings
        that we simulate when the update is not automatically accepted

    They refers to each of the outcomes of Metropolis-Hastings that we want to test:
    - accepted
    - accepted when the random sampled acceptance is smaller than the actual one
    - rejected when the random sampled acceptance is bigger than the actual one
    """

    return [{'t_gen': 0.3, 't_rem': 0.5, 'actual_acceptance': 1.0},
    {'t_gen': 0.6, 't_rem': 0.8, 'actual_acceptance': 0.68, 'sampled_acceptance' : 0.67},
    {'t_gen': 0.6, 't_rem': 0.8, 'actual_acceptance': 0.68, 'sampled_acceptance' : 0.69}]

def test_eval_add_internal_accepted(polaron, parameters_eval_add_internal):
    """This test checks the behaviour of the eval_add_internal method when:
    - acceptance probability for the update is 1.0
    - update is accepted directly
     
    The eval_add_internal function is responsible for:
    - evaluating the acceptance probability for the addition of a specific phonon
    - add it into the current diagram for this outcome of Metropolis-Hastings 
      because the acceptance probability is 1.0

    The test involves patching:
    - numpy.random.uniform to control the values of the random sampled
      times of the phonon 

    This is fundamental to calculate manually the actual acceptance probability
    for the update and to test that the updated state of the object is the 
    expected one after the call to the function.

    Parameters:
        polaron: polaron fixture with zero order
        parameters_eval_add_internal: list of dictionaries with times of the
            phonon and relevant values for acceptance probabilities

    GIVEN: a polaron with order zero, two specific times for the generated phonon,
      the acceptance probability for its addition given by Metropolis-Hastings of 1.0
    WHAT: apply the eval_add_internal function patching 
      the random times of the phonon
    THEN: get a polaron object with one more phonon and order increased by two
    """
    # Get the initial state
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_order = polaron.diagram['order']

    with patch('numpy.random.uniform') as mock_uniform:
        mock_uniform.side_effect = [parameters_eval_add_internal[0]['t_gen'], 
                        parameters_eval_add_internal[0]['t_rem']]
        # Call the method under test
        polaron.eval_add_internal()

    assert len(polaron.phonon_list) == initial_phonon_list_length + 1
    assert polaron.diagram['order'] == initial_order + 2


def test_eval_add_internal_accepted_conditional(polaron, parameters_eval_add_internal):
    """This test checks the behaviour of the eval_add_internal method when:
    - acceptance probability for the update is less than 1.0
    - update is accepted after sampling an acceptance probability smaller
      than the one returned by Metropolis-Hastings.
     
    The eval_add_internal function is responsible for:
    - evaluating the acceptance probability for the addition of a specific phonon
    - add it into the current diagram for this outcome of Metropolis-Hastings 
      because we simulate the sampling of an acceptance probability smaller
      than the one returned by Metropolis-Hastings.

    The test involves patching:
    - numpy.random.uniform to control the values of the random sampled
      times of the phonon and of the sampled acceptance probability 

    This is fundamental to calculate manually the actual acceptance probability
    for the update and to test that the updated state of the object is the 
    expected one after the call to the function.

    Parameters:
        polaron: polaron fixture with zero order
        parameters_eval_add_internal: list of dictionaries with times of the
            phonon and relevant values for acceptance probabilities

    GIVEN: a polaron with order zero, two specific times for the generated phonon,
      the acceptance probability for its addition given by Metropolis-Hastings of 0.54
      and a sampled acceptance probability of 0.53
    WHAT: apply the eval_add_internal function patching the random times of 
      the phonon and the random acceptance probability
    THEN: get a polaron object with one more phonon and order increased by two
    """
    # Get the initial state
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_order = polaron.diagram['order']
    
    with patch('numpy.random.uniform') as mock_uniform:
        mock_uniform.side_effect = [parameters_eval_add_internal[1]['t_gen'], 
                        parameters_eval_add_internal[1]['t_rem'], 
                        parameters_eval_add_internal[1]['sampled_acceptance']]
        polaron.eval_add_internal()

    assert len(polaron.phonon_list) == initial_phonon_list_length + 1
    assert polaron.diagram['order'] == initial_order + 2

def test_eval_add_internal_rejected(polaron, parameters_eval_add_internal):
    """This test checks the behaviour of the eval_add_internal method when:
    - acceptance probability for the update is less than 1.0
    - update is rejected after sampling an acceptance probability bigger
      than the one returned by Metropolis-Hastings.
     
    The eval_add_internal function is responsible for:
    - evaluating the acceptance probability for the addition of a specific phonon
    - leave unaltered the current diagram for this outcome of Metropolis-Hastings 
      because we simulate the sampling of an acceptance probability bigger
      than the one returned by Metropolis-Hastings.

    The test involves patching:
    - numpy.random.uniform to control the values of the random sampled
      times of the phonon and of the sampled acceptance probability 

    This is fundamental to calculate manually the actual acceptance probability
    for the update and to test that the updated state of the object is the 
    expected one after the call to the function.

    Parameters:
        polaron: polaron fixture with zero order
        parameters_eval_add_internal: list of dictionaries with times of the
            phonon and relevant values for acceptance probabilities

    GIVEN: a polaron with order zero, two specific times for the generated phonon,
      the acceptance probability for its addition given by Metropolis-Hastings of 0.54
      and a sampled acceptance probability of 0.55
    WHAT: apply the eval_add_internal function patching the random times of 
      the phonon and the random acceptance probability
    THEN: get a polaron object with the same order and number of phonons
    """
    # Get the initial state
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_order = polaron.diagram['order']

    with patch('numpy.random.uniform') as mock_uniform:
        mock_uniform.side_effect = [parameters_eval_add_internal[2]['t_gen'], 
                        parameters_eval_add_internal[2]['t_rem'], 
                        parameters_eval_add_internal[2]['sampled_acceptance']]
        polaron.eval_add_internal()

    assert len(polaron.phonon_list) == initial_phonon_list_length
    assert polaron.diagram['order'] == initial_order

@pytest.fixture
def polaron_order_four(phonon, another_phonon):
    """This method return a fixed polaron object
    with two phonons that can be used consistently
    during the tests
    
    Parameters:
        phonon: fixture phonon with specific times
        another_phonon: fixture another_phonon with specific times
    
    Return:
        polaron: polaron object with an updated phonon list
            that contains phonon and another_phonon and 
            a diagram order of four
    """
    polaron = Polaron(omega=1.0, g=0.5, time=10.0)
    polaron.add_internal(phonon)
    polaron.add_internal(another_phonon)
    return polaron

def test_weigth_ratio_remove(polaron_order_two, phonon):
    """This test checks whether the ratio between a diagram
    with one less phonon and the current one gives the expected value

    Parameters:
        polaron_order_two: fixture polaron_order_two
        phonon: fixture phonon
    
    GIVEN: a polaron object of order two with
      specific parameters and a phonon with specific times
    WHAT: apply the weight_ratio_remove to evaluate the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula
    """
    ratio = polaron_order_two.weigth_ratio_remove(phonon)
    expected_ratio =  np.exp(10.0 * 1.0 * (0.5 - 0.2)) / ((0.5 * 10.0) ** 2)
    assert isclose(ratio,expected_ratio)


def test_proposal_remove_ratio_order_two(polaron_order_two, phonon):
    """This test checks whether the ratio between the proposal
    probabilities of the reverse and direct process for the 
    remove_phonon update gives the expected value for a 
    polaron of order two

    Parameters:
        polaron_order_two: fixture polaron_order_two
        phonon: fixture phonon
    
    GIVEN: a polaron object of order two and a phonon 
        with specific times
    WHAT: apply the function proposal_remove_ratio to evaluate 
        the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula
    """
    ratio = polaron_order_two.proposal_remove_ratio(phonon)
    expected_ratio = 2 * 1/(1 - 0.2)
    assert isclose(ratio,expected_ratio)

def test_proposal_remove_ratio_order_greater_two(polaron_order_four, phonon):
    """This test checks whether the ratio between the proposal
    probabilities of the reverse and direct process for the 
    remove_phonon update gives the expected value using the 
    formula valid for polarons of order grater than two

    Parameters:
        polaron_order_four: fixture polaron_order_four
        phonon: fixture phonon
    
    GIVEN: a polaron object of order greater than two
        and a phonon with specific times
    WHAT: apply the function proposal_remove_ratio to evaluate 
        the ratio
    THEN: the ratio has to be equal to the expected ratio 
        evaluated through the formula
    """
    ratio = polaron_order_four.proposal_remove_ratio(phonon)
    expected_ratio = (len(polaron_order_four.phonon_list))/(1 - 0.2)
    assert isclose(ratio,expected_ratio)

@pytest.mark.parametrize("phonon_index", [0,1])
def test_choose_phonon(polaron_order_four, phonon_index):
    """This test checks whether the choose_phonon 
    function retrieves an index from the ones
    in phonon_list.

    This tests mocks the random sampling of integers 
    from np.random.randint
    
    Parameters:
        polaron_order_four: fixture polaron_order_four
        phonon_index: integer corresponding to the index 
            of the phonon to select in the phonons list

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

    assert actual_index == phonon_index

@pytest.mark.parametrize("parameters", [
    {'polaron': "polaron_order_four", 'phonon_index': 0},
    {'polaron': "polaron_order_two", 'phonon_index': 0}])
def test_remove_internal(parameters, request):
    """This test checks that the remove_internal method
    removes a phonon from the phonon_list and updates the order

    Parameters:
        parameters: dictionary with the polaron to consider 
            for the update and the phonon index
        request: pytest fixture used to access the value of a fixture
            having its name passed as a string
    
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

@pytest.fixture
def parameters_eval_remove_internal(polaron_order_two, polaron_order_four):
    """This fixture returns a list of dictionaries where each contains the
    relevant structure and internal parameters used in the tests of
    eval_remove_internal:

     1) polaron object to apply eval_remove_internal to 
     2) a phonon index that refers to the phonon to remove
     3) the acceptance probability that the remove_internal update is accepted.
        It is evaluated manually for the selected phonon and polaron
     4) the sampled acceptance probability from Metropolis-Hastings
        that we simulate when the update is not automatically accepted

    Each dictionary refers to one of the outcomes of Metropolis-Hastings 
    that we want to test for the remove_internal update:
    - accepted
    - accepted when the random sampled acceptance is smaller than the actual one
    - rejected when the random sampled acceptance is bigger than the actual one

    Parameters:
        polaron_order_two: fixture polaron_order_two
        polaron_order_four: fixture polaron_order_four
    """
    return [{'polaron': polaron_order_two, 'phonon_index' : 0, 'actual_acceptance': 1.0},
    {'polaron': polaron_order_four, 'phonon_index' : 1, 'actual_acceptance': 0.54, 'sampled_acceptance' : 0.53},
    {'polaron': polaron_order_four, 'phonon_index' : 1, 'actual_acceptance': 0.54, 'sampled_acceptance' : 0.55}]

def test_eval_remove_internal_accepted(parameters_eval_remove_internal):
    """This test checks the behaviour of the eval_remove_internal method when:
    - acceptance probability for the update is 1.0
    - update is accepted directly
     
    The eval_remove_internal function is responsible for:
    - evaluating the acceptance probability for the removal of a specific phonon
    - remove it from the current diagram for this outcome of Metropolis-Hastings 
      because the acceptance probability is 1.0

    The test involves patching:
    -numpy.random.integer to control the value of random integer
      that corresponds to the index of the phonon to remove

    This is fundamental to calculate manually the acceptance probability
    for the update and to test that the update state of the object is the 
    expected one after the call to the function.

    Parameters:
        parameters_eval_remove_internal: list of dictionaries with polarons,
            phonon_index and relevant values for acceptance probabilities

    GIVEN: a polaron with order two and a phonon for which
      the acceptance probability given by Metropolis-Hastings is 1.0
    WHAT: apply the eval_remove_internal function patching the random 
      phonon index
    THEN: get a polaron object with one less phonon and order decreased by two
    """
    # Get the initial state
    polaron=parameters_eval_remove_internal[0]['polaron']
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_order = polaron.diagram['order']

    with patch('numpy.random.randint') as mock_randint:
        mock_randint.return_value = parameters_eval_remove_internal[0]['phonon_index']
        # Call the method under test
        polaron.eval_remove_internal()

        assert len(polaron.phonon_list) == initial_phonon_list_length - 1
        assert polaron.diagram['order'] == initial_order - 2

def test_eval_remove_internal_accepted_conditional(parameters_eval_remove_internal):
    """This test checks the behaviour of the eval_remove_internal method when:
    - acceptance probability for the update is less than 1.0
    - update is accepted after sampling an acceptance probability smaller
      than the one returned by Metropolis-Hastings.
     
    The eval_remove_internal function is responsible for:
    - evaluating the acceptance probability for the removal of a specific phonon
      randomly chosen
    - remove it into the current diagram for this outcome of Metropolis-Hastings 
      because we simulate the sampling of an acceptance probability smaller
      than the one returned by Metropolis-Hastings.

    The test involves patching:
    -numpy.random.integer to control the value of random integer
      that corresponds to the index of the phonon to remove
    -numpy.random.uniform to control the value of the random sampled
      acceptance probability 

    This is fundamental to calculate manually the acceptance probability
    for the update and to test that the updated state of the object is the 
    expected one after the call to the function.

    Parameters:
        parameters_eval_remove_internal: list of dictionaries with polarons,
            phonon_index and relevant values for acceptance probabilities

    GIVEN: a polaron with order four, a phonon for which
      the acceptance probability given by Metropolis-Hastings is 0.54
      and a sampled acceptance probability of 0.53
    WHAT: apply the eval_remove_internal function patching the random 
      phonon index and the random acceptance probability
    THEN: get a polaron object with one less phonon and order decreased by two
    """
    # Get the initial state
    polaron=parameters_eval_remove_internal[1]['polaron']
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_order = polaron.diagram['order']
 
    with patch('numpy.random.randint') as mock_randint, \
        patch('numpy.random.uniform') as mock_uniform:

        mock_randint.return_value = parameters_eval_remove_internal[1]['phonon_index']
        mock_uniform.return_value = parameters_eval_remove_internal[1]['sampled_acceptance']

        polaron.eval_remove_internal()

        assert len(polaron.phonon_list) == initial_phonon_list_length - 1
        assert polaron.diagram['order'] == initial_order - 2

def test_eval_remove_internal_rejected(parameters_eval_remove_internal):
    """This test checks the behaviour of the eval_remove_internal method when:
    - acceptance probability for the update is less than 1.0
    - update is rejected after sampling an acceptance probability bigger
      than the one returned by Metropolis-Hastings.
     
    The eval_remove_internal function is responsible for:
    - evaluating the acceptance probability for the removal of a specific phonon
      randomly chosen
    - leave unaltered the current diagram for this outcome of Metropolis-Hastings 
      because we simulate the sampling of an acceptance probability bigger
      than the one returned by Metropolis-Hastings.

    The test involves patching:
    -numpy.random.integer to control the value of random integer
      that corresponds to the index of the phonon to remove
    -numpy.random.uniform to control the value of the random sampled
      acceptance probability 

    This is fundamental to calculate manually the acceptance probability
    for the update and to test that the state of the object is unaltered
    after the call to the function.

    Parameters:
        parameters_eval_remove_internal: list of dictionaries with polarons,
            phonon_index and relevant values for acceptance probabilities

    GIVEN: a polaron with order four, a phonon for which
      the acceptance probability given by Metropolis-Hastings is 0.54
      and a sampled acceptance probability of 0.55
    WHAT: apply the eval_remove_internal function patching the random 
      phonon index and the random acceptance probability
    THEN: get a polaron object with the same order and number of phonons
    """
    # Get the initial state
    polaron=parameters_eval_remove_internal[2]['polaron']
    initial_phonon_list_length = len(polaron.phonon_list)
    initial_order = polaron.diagram['order']

    #mocking functions for phonon index and sampled acceptance
    with patch('numpy.random.randint') as mock_randint, \
        patch('numpy.random.uniform') as mock_uniform:

        mock_randint.return_value = parameters_eval_remove_internal[2]['phonon_index']
        mock_uniform.return_value = parameters_eval_remove_internal[2]['sampled_acceptance']

        polaron.eval_remove_internal()

        assert len(polaron.phonon_list) == initial_phonon_list_length
        assert polaron.diagram['order'] == initial_order

def test_eval_diagram_energy_zero_order(polaron):
    """This test checks that the energy of a Polaron object
    with zero order evaluated at a MonteCarlo step is zero as 
    defined by the estimator formula

    Parameters:
        polaron: fixture polaron 

    GIVEN: a Polaron object with zero order
    WHAT: call the eval_diagram_energy function
    THEN: the energy of the polaron has to be zero
        consistently with the estimator
    """
    polaron.eval_diagram_energy()
    assert isclose(polaron.diagram['total_energy'],0.0)

def test_eval_diagram_energy_order_two(polaron_order_two):
    """This test checks that the energy of a Polaron object
    with order greater than zero evaluated at a MonteCarlo step 
    takes the value evaluated by the estimator formula

    Parameters:
        polaron_order_two: fixture polaron_order_two

    GIVEN: a Polaron object with order two
    WHAT: call the eval_diagram_energy function
    THEN: the energy of the polaron has to be
        consistent with the one of the estimator
    """
    assert polaron_order_two.diagram['order'] == 2
    phonons_interaction_time_sum = 0.5 - 0.2
    actual_energy = 1.0 * phonons_interaction_time_sum - 2 / 10.0
    #calls to the function eval_diagram_energy
    polaron_order_two.eval_diagram_energy()
    assert isclose(polaron_order_two.diagram['total_energy'],actual_energy)

def test_update_diagrams_info(polaron_order_two):
    """This test checks that the lists with diagrams order and energy
       of the polaron are updated correctly during the simulation.

       Parameters:
            polaron_order_two: fixture polaron_order_two

       Notes:
        The values for order and energy of a Polaron with zero order
        are inserted as initial values in the lists order_sequence and
        energy_sequence. After the update is checked that the actual lists 
        are equal to the expected ones
       
       GIVEN: a polaron object of order two
       WHAT : mock some initial values and then
            apply update_diagrams_info function
       THEN : have two lists with the correct 
            values for order and energy
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
