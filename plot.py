import numpy as np
import matplotlib.pyplot as plt
from os import stat, path
from matplotlib.lines import Line2D
from scipy.optimize import curve_fit
from scipy.stats import poisson

def get_bins_edges(sequence : np.ndarray):
    """This method set the left and right edges for each bin of
    the histogram. The bin is centered on each of the possible integer
    values sampled

    Parameters:
        sequence : an array that contains the sampled values
    
    Return:
        left and right edges for each bin in the histogram
    """
    n_bins = np.max(sequence) + 1
    bins_edges = [i-0.5 for i in range(n_bins+1)]
    return bins_edges

def poisson_func(mean, par):
    "This function return a poisson variable "
    return poisson.pmf(mean, par)

def plot_montecarlo(order_sequence : list, energy_sequence : list, 
                    settings : dict[str, str], path_plot : dict[str, str],
                    path_data : dict[str,str]):
    """
    This function:
    - ESTIMATES the ground state energy of the polaron through the corresponding
    estimator
    - PLOTS an histogram of the phonons distribution sampled during the MonteCarlo 
    simulation and saves the corresponding picture.
    - PRINT on a file the relevant parameters for the simulation
    
    Parameters:
        -order_sequence: sequence of orders of each of the sampled diagrams in the simulation
        -energy_sequence: sequence of energies of each of the sampled diagrams in the simulation
        -settings : dictionary with the physical and technical parameters of the simulation
            (e.g. frequency of phonons, number of MonteCarlo steps...)
        -path_plot: dictionary with the destinations to store plots (e.g. phonons distribution)
        -path_data: dictionary with the destinations to store relevant parameters and
            output of the simulation (e.g. ground state energy and mean number of phonons given a set
            of initial physical parameters)
    ---------------------------
    Notes:
    -Occurrences for each bin of the histogram are normalized to the number of samples
    -Plot of the sampled probability distribution vs the analitycal one (Poisson)
    """

    energy_sequence=np.array(energy_sequence)
    phonons_sequence = np.divide(order_sequence,2).astype('uint32')

    mean_phonons = np.mean(phonons_sequence)
    mean_energy = np.mean(energy_sequence)

    #variable from settings dictionary
    g = settings['G']
    omega = settings['OMEGA']
    time = settings['TIME']
    nsteps = settings['NSTEPS']
    nsteps_burn = settings['NSTEPS_BURN']
    interactive = settings['INTERACTIVE']

    #path plot
    plot_phonons = path_plot['PHONONS']

    #path data
    data_energy_phonons = path_data['ENERGY+PHONONS']
    append_data = path_data['APPEND']
    
    energy_label = r'$E_{GS} =$ ' + f'{mean_energy:.3f}'
    phonons_label = 'DiagMC: ' + r'$\overline{N_{pho}}/\omega\tau$ = ' + f'{mean_phonons/(time*omega):.4f}'

    occ, bins, _ = plt.hist(phonons_sequence, density=True, bins=get_bins_edges(phonons_sequence),
        ec='black', fc='blue', alpha=0.8)
    
    bins_center = np.array([(a+b)/2 for a,b in zip(bins[:-1], bins[1:])])
    par, _ = curve_fit(poisson_func, bins_center, occ, maxfev=5000)
    x_phonons = np.arange(phonons_sequence.min()-5,phonons_sequence.max()+5)
    plt.plot(x_phonons, poisson_func(x_phonons, *par), color='red', ls='-')
    plt.xlabel(r'$N_{phonons}$')
    plt.ylabel(r'P($N_{phonons}$)')
    
    legend_elements = [Line2D([0], [0], color='b', label=phonons_label),
                        Line2D([0], [0], color='r', label=r'Poisson fit: ' + r'$\overline{N_{pho}}/\omega\tau = $ ' + 
                               '%5.4f' % tuple(par/(omega*time))),
                        Line2D([0], [0], color='y', label=r'Theory: ' + r'$N_{pho} = $ ' + 
                               r'$\frac{g^{2}}{\omega^{2}}$ = ' + f'{g**2/omega**2:.4f}'),
                        Line2D([0], [0], color='g', label=energy_label)]
    plt.legend(handles=legend_elements)
    
    plt.suptitle(fr'''g = {g} $\omega$ = {omega} nsteps_burn = {nsteps_burn} nsteps = {nsteps} $\tau$ = {time}''')
    plt.savefig(plot_phonons, bbox_inches='tight')
    if interactive :
        plt.show()

    if not path.isfile(data_energy_phonons):
        with open(data_energy_phonons, 'a') as file:
            pass

    #if file empty write column labels and values
    if stat(data_energy_phonons).st_size == 0 or not append_data:      
        with open(data_energy_phonons, 'w') as file:
            #column labels for output 
            print('g' + ' ' + 'omega' + ' ' + 'time' + ' ' + 
                  'nsteps_burn' + ' ' + 'nsteps' + ' ' + 
                  'mean_phonons_DMC' + ' ' + 'mean_energy_DMC', 
                  file=file)
            #actual values
            print(str(g) + ' ' + str(omega) + ' ' + 
                  str(time) + ' ' + str(nsteps_burn) + ' ' + 
                  str(nsteps) + ' ' + f'{mean_phonons:.5f}' + ' ' +
                  f'{mean_energy:.5f}', file=file)
    else:
        #print actual values
        with open(data_energy_phonons, 'a') as file:
            #actual values
            print(str(g) + ' ' + str(omega) + ' ' + 
                  str(time) + ' ' + str(nsteps_burn) + ' ' + 
                  str(nsteps) + ' ' + f'{mean_phonons:.5f}' + ' ' +
                  f'{mean_energy:.5f}', file=file)



