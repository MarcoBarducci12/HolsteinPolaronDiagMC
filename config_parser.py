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
        """This function stores in a vocabulary the physical and technical 
        parameters for the simulation
        
        Return:
            vocabulary with relevant parameters for the simulation
        """
        return {'NSTEPS' : int(self.config.get('settings', 'NSTEPS')),
                'NSTEPS_BURN' : int(self.config.get('settings', 'NSTEPS_BURN')),
                'OMEGA' : float(self.config.get('settings', 'OMEGA')),
                'G' : float(self.config.get('settings', 'G')),
                'TIME' : float(self.config.get('settings', 'TIME')),
                'INTERACTIVE' : self.config.getboolean('settings', 'INTERACTIVE')}
    
    def get_path_plot(self):
        """This function stores in a vocabulary the path to save plots
        
        Return:
            vocabulary that stores str identifiers and the path to store
            the corresponding plot
        """
        return {'PLOT_FOLDER' : self.config.get('path_plot', 'PLOT_FOLDER'),
                'PHONONS' : self.config.get('path_plot', 'PHONONS')}
    
    def get_path_data(self):
        """This function stores in a vocabulary the path and features 
        to save relevant data from the simulation
        
        Return:
            vocabulary that stores str identifiers path and mode to store
            the corresponding data
        """
        return {'DATA_FOLDER' : self.config.get('path_data', 'DATA_FOLDER'), 
                'ENERGY+PHONONS' : self.config.get('path_data', 'ENERGY+PHONONS'),
                'APPEND' : self.config.getboolean('path_data', 'APPEND')}

def check_positive_parameters(settings):
    """This function checks that the relevant parameters for the simulation hold positive values
        otherwise we would have a meaningless physical picture and without any steps the
        simulation would not take place
    
        Parameters:
            settings: vocabulary with the relevant parameters for the simulation
    """
    if settings['NSTEPS'] <= 0:
        raise ValueError(f'The number of MonteCarlo steps must be > 0 but is {settings["""NSTEPS"""]}')
    elif settings['OMEGA'] <= 0.0:
        raise ValueError(f'The phonon frequency must be > 0.0 but is {settings["""OMEGA"""]}')
    elif settings['G'] <= 0.0:
        raise ValueError(f'The intensity of electron phonon coupling must be > 0.0 but is {settings["""G"""]}')
    elif settings['TIME'] <= 0.0:
        raise ValueError(f'The lifetime of the electron must be > 0.0 but is {settings["""TIME"""]}')

def check_storage_directories_exist(path_plot, path_data):
    """This function checks that the destination path to store plot and data
        exist otherwise it creates them 
    
        Parameters:
            path_plot: vocabulary with the path to store plot
            path_data: vocabulary with the path to store data
    """
    #create path for plot folder
    if not path.exists(path_plot['PLOT_FOLDER']):
        makedirs(path_plot['PLOT_FOLDER'])
    if not path.exists(path_data['DATA_FOLDER']):
        makedirs(path_data['DATA_FOLDER'])

