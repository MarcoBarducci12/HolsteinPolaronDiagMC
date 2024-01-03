import pytest
from os import path
from configparser import ConfigParser
from config_parser import (check_positive_parameters, 
                        check_storage_directories_exist, 
                        Config)

def test_config_initialization():
    """This test checks whether the Config class
     is initialized correctly holding a ConfigParser object
     
    GIVEN: the name of the configuration file
    WHEN: call the constructor of Config class
    THEN: the resulting config object should be of type
            ConfigParser
    """
    
    config = Config('configuration.txt')
    assert isinstance(config.config, ConfigParser)

def test_config_keys_initialization():
    """This test checks whether the get_settings, 
    get_path_plot, get_path_data methods return dictionaries
    with the expected keys
     
    GIVEN: a valid config object
    WHEN: apply the get_settings(), get_path_plot(), get_data_plot()
    THEN: the resulting dictionaries contain the expected keys
    """

    config = Config('configuration.txt')
    settings = config.get_settings()
    path_plot = config.get_path_plot()
    path_data = config.get_path_data()

    assert 'NSTEPS' in settings
    assert 'NSTEPS_BURN' in settings
    assert 'G' in settings
    assert 'OMEGA' in settings
    assert 'TIME' in settings
    assert 'INTERACTIVE' in settings

    assert 'PLOT_FOLDER' in path_plot
    assert 'PHONONS' in path_plot

    assert 'DATA_FOLDER' in path_data
    assert 'ENERGY+PHONONS' in path_data
    assert 'APPEND' in path_data

def test_check_positive_parameters():
    """This test checks whether the function check_positive_parameters
      raises a ValueError for invalid settings and does not raise
      an error for valid settings
      
    GIVEN: an invalid (valid) settings of parameters
    WHAT: apply to it the check_positive_parameters function
    THEN: it raises (not raises) a ValueError
    """
    
    invalid_settings = {'NSTEPS' : 0, 'OMEGA' : -1.0, 'G' : 1.0, 'TIME' : -10.0}
    with pytest.raises(ValueError):
        check_positive_parameters(invalid_settings)

    valid_settings = {'NSTEPS' : 100, 'OMEGA' : 1.0, 'G' : 1.0, 'TIME' : 10.0}
    check_positive_parameters(valid_settings)

def test_check_storage_directories_exist(tmp_path):
    """This test tests the behaviour of check_storage_directories_exist
    function which checks whether the provided directories exists and
    it creates them if they do not exist
    
    Parameters:
        tmp_path : temporary directory path provided by Pytest

    GIVEN: a temporary directory path
    WHAT: apply to it check_storage_directories_exist function
    THEN: create the directories if they do no exist
    """

    path_plot = {'PLOT_FOLDER': str(tmp_path / 'plot/')}
    path_data = {'DATA_FOLDER': str(tmp_path / 'data/')}

    check_storage_directories_exist(path_plot, path_data)

    assert path.exists(path_plot['PLOT_FOLDER'])
    assert path.exists(path_data['DATA_FOLDER'])