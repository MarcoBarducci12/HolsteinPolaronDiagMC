import pytest
from os import path, rmdir
from math import isclose
from configparser import ConfigParser
from config_parser import (check_positive_parameters, 
                        ensure_storage_directories_exist, 
                        Config)

def test_config_initialization():
    """This test checks whether the Config class
     is initialized correctly holding a ConfigParser object
     
    GIVEN: the name of the configuration file
    WHEN: call the constructor of Config class
    THEN: the resulting config object should be of type
            ConfigParser
    """
    
    config = Config('configuration_mock.txt')
    assert isinstance(config.config, ConfigParser)

def test_config_keys_values_initialization():
    """This test checks whether the get_settings, get_seed 
    get_path_plot, get_path_data methods return dictionaries
    with the expected keys and values
     
    GIVEN: a valid config object
    WHEN: apply the get_settings(), get_path_plot(), get_data_plot()
    THEN: the resulting dictionaries contain the expected keys and values
    """

    config = Config('configuration_mock.txt')
    settings = config.get_settings()
    seed_dict = config.get_seed()
    path_plot = config.get_path_plot()
    path_data = config.get_path_data()

    assert settings['NSTEPS'] == 100000
    assert settings['NSTEPS_BURN'] == 1000
    assert isclose(settings['G'],0.5)
    assert isclose(settings['OMEGA'],1.0)
    assert isclose(settings['TIME'],50.0)
    assert settings['INTERACTIVE'] == False
    assert seed_dict['SEED'] == None

    assert path_plot['PLOT_FOLDER'] == "./plot"
    assert path_plot['PHONONS'] == ("./plot/phonons_distribution_g_0.5"
                                    "_omega_1.0_time_50.0_nsteps_100000"
                                    "_nsteps_burn_1000.png")

    assert path_data['DATA_FOLDER'] == "./data"
    assert path_data['ENERGY+PHONONS'] == "./data/energy_phonons.txt"
    assert path_data['APPEND'] == False

def test_check_positive_parameters():
    """This test checks whether the function check_positive_parameters
      raises a ValueError for invalid settings
      
    GIVEN: an invalid settings of parameters
    WHAT: apply to it the check_positive_parameters function
    THEN: it raises a ValueError
    """
    
    invalid_settings = {'NSTEPS' : 0, 'OMEGA' : -1.0, 'G' : 1.0, 'TIME' : -10.0}
    with pytest.raises(ValueError):
        check_positive_parameters(invalid_settings)

def test_ensure_storage_directories_exist():
    """This test tests the behaviour of ensure_storage_directories_exist
    function which ensures that the target directories exists either
    creating them if absents or leaving the directories unaltered
    if they already exist

    GIVEN: test directories paths
    WHAT: apply to it ensures_storage_directories_exist function
    THEN: create the directories if they do no exist
    """

    path_plot = {'PLOT_FOLDER': str('mock_plots/')}
    path_data = {'DATA_FOLDER': str('mock_data/')}

    ensure_storage_directories_exist(path_plot, path_data)

    assert path.exists(path_plot['PLOT_FOLDER'])
    assert path.exists(path_data['DATA_FOLDER'])

    rmdir(path_plot['PLOT_FOLDER'])
    rmdir(path_data['DATA_FOLDER'])

    assert not path.exists(path_plot['PLOT_FOLDER'])
    assert not path.exists(path_data['DATA_FOLDER'])