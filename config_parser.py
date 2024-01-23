import configparser
from os import path, makedirs

class Config:
    def __init__(self, filename : str):
        """This function creates a ConfigParser object that can read
        data from a configuration file
        
        Parameters:
            filename: name of the configuration file with the values
        """
        self.config = configparser.ConfigParser(interpolation=configparser.ExtendedInterpolation())
        self.config.read(filename)

    def get_settings(self):
        """This function stores in a dictionary the physical and technical 
        parameters for the simulation
        
        Return:
            dictionary with relevant parameters for the simulation
        """
        return {'NSTEPS' : int(self.config.get('settings', 'NSTEPS')),
                'NSTEPS_BURN' : int(self.config.get('settings', 'NSTEPS_BURN')),
                'OMEGA' : float(self.config.get('settings', 'OMEGA')),
                'G' : float(self.config.get('settings', 'G')),
                'TIME' : float(self.config.get('settings', 'TIME')),
                'INTERACTIVE' : self.config.getboolean('settings', 'INTERACTIVE')}
    
    def get_path_plot(self):
        """This function stores in a dictionary the path to save plots
        
        Return:
            dictionary that stores str identifiers and the path to store
            the corresponding plot
        """
        return {'PLOT_FOLDER' : self.config.get('path_plot', 'PLOT_FOLDER'),
                'PHONONS' : self.config.get('path_plot', 'PHONONS')}
    
    def get_path_data(self):
        """This function stores in a dictionary the path and features 
        to save relevant data from the simulation
        
        Return:
            dictionary that stores str identifiers path and mode to store
            the corresponding data
        """
        return {'DATA_FOLDER' : self.config.get('path_data', 'DATA_FOLDER'), 
                'ENERGY+PHONONS' : self.config.get('path_data', 'ENERGY+PHONONS'),
                'APPEND' : self.config.getboolean('path_data', 'APPEND')}

def check_positive_parameters(settings):
    """This function checks that the relevant parameters for the simulation are positive.
       
       It collects all the strings relative to invalid parameters that would lead to a
       nonsensical simulation if presents and raise a ValueError stating all the invalid ones.
    
        Parameters:
            settings: dictionary that contains relevant parameters for the simulation
        
        Raises:
            ValueError: if any of the following conditions are met:
                - NSTEPS : not greater than 0
                - OMEGA (phonon frequency) : not greater than 0.0
                - G (intensity e-ph coupling) : not greater than 0.0
                - TIME (lifetime of the electron) : not greater than 0.0
    """
    invalid_parameters = []
    if settings['NSTEPS'] <= 0:
        invalid_parameters.append(f'The number of MonteCarlo steps must be > 0 but is {settings["""NSTEPS"""]}')
    if settings['OMEGA'] <= 0.0:
        invalid_parameters.append(f'The phonon frequency must be > 0.0 but is {settings["""OMEGA"""]}')
    if settings['G'] <= 0.0:
        invalid_parameters.append(f'The intensity of electron phonon coupling must be > 0.0 but is {settings["""G"""]}')
    if settings['TIME'] <= 0.0:
        invalid_parameters.append(f'The lifetime of the electron must be > 0.0 but is {settings["""TIME"""]}')
    
    if invalid_parameters:
        raise ValueError('\n'.join(invalid_parameters))

def ensure_storage_directories_exist(path_plot, path_data):
    """This function ensures that the target directories to store
      plot and data exist. It either creates them if they are absent or 
      leaves the target directories unaltered if already present
    
        Parameters:
            path_plot: dictionary with the path to store plot
            path_data: dictionary with the path to store data
    """
    makedirs(path_plot['PLOT_FOLDER'], exist_ok=True)
    makedirs(path_data['DATA_FOLDER'], exist_ok=True)

