# Holstein Polaron Diagrammatic MonteCarlo

This program finds the ground state energy $E_{GS}$ of the Holstein hamiltonian
$$H = \varepsilon\hat{c}^{\dag}\hat{c} + \omega\hat{b}^{\dag}\hat{b} + g(\hat{b}^{\dag} + \hat{b})\,\hat{c}^{\dag}\hat{c}$$
which describes a single electron with energy $\varepsilon$ in an atomic orbital  that couples via a vertex interaction of intensity $g$ to phonons of the lattice with frequency $\omega$.
This hamiltonian has an analytic solution both for $E_{GS}$, the mean number of phonons $N_{pho}$ and the phonon distribution $P(n)$
$$E_{GS} = -\frac{g^{2}}{\omega}$$
$$N_{pho} = \frac{g^{2}}{\omega^{2}} \quad \textrm{and} \quad P(n) = e^{N_{pho}}\,\frac{1}{n!}\,(N_{pho})^{n}$$

In this simulation we use a Markov Chain MonteCarlo algorithm to sample Feynman diagrams obtained in the series expansion of the Green's function. The simulation proceeds selecting randomly at each MonteCarlo step one of the possible updates for the current diagram which are _add\_phonon_ and _remove\_phonon_ and accept or reject it based on the outcome of Metropolis-Hastings criterion.

At each step we collect the order and evaluate the energy of the current diagram via the use of an estimator which averaged over the MonteCarlo samples retrieves $E_{GS}$
$$\left\langle \frac{\sum_{i}^{pho} \omega \Delta\tau_{i} - \textrm{order}}{\tau}\right\rangle_{\textrm{DMC}}$$ 
the label _order_ in the program refers to the number of e-ph vertexes in the diagram instead of the actual order of the diagram in perturbation theory. 

Moreover, since we consider only diagrams where the electron is coupled to _internal_ phonons i.e. curly interactions lines whose both extrema have an intersection with the line of the electron it means that $order = 2* \# phonons$.

For this reason collecting the order of the sampled diagrams let us evaluate $N_{pho}$ and $P(n)$ and plot the histogram for $P(n)$ just after dividing by 2 each occurrence for the order of a diagram. 

## User guide

### Installation
To install the program clone the repository, this will create a folder `HolsteinDiagMC` which is the root of the project. You will need a working `python3` environment to run the code.

Run ```pip3 install -r requirements.txt```
to check whether you have the required packages to run the simulation. If some of them are missing they will be installed

### How to run the program
After checking the requirements you can run the program executing the following command
```python3 main.py configuration.txt```
The program reads the main parameters for the simulation from a configuration file called _configuration.txt_ that must be provided as a command line argument.

The `[settings]` section contains some physical parameters such as:
- g: electron-phonon coupling
- omega : frequency of phonons
and some technical parameters like:
- nsteps: number of MonteCarlo steps
- nsteps_burn: number of steps for thermalization

the `[path_plot]` and `[path_data]` contains information about the path name where the image of the plot and the data are saved


### Structure of the project

`config_parser.py`

The program uses a custom `Config` object defined in the module `config_parser.py` to read the simulation parameters from a configuration file named _configuration.txt_. It is a wrapper around a `ConfigParser` object from `configparser` that implements a few function to handle the simulation parameters in a compact way using python dictionaries

`polaron.py`

The core of the program is the `polaron.py` module. Here we defined a Polaron class that stores in the attribute `diagram` the simulation parameters and the information of the current diagram i.e. *order* and *phonon_list* which represent the number of vertexes in the diagram and a list of the phonons coupled to the electrons. Each phonon is an array of two times: respectively the istant when the phonon couples with the electron and its removal time.
Moreover, the Polaron class contains as attributes two lists: `order_sequence` and `energy_sequence` to store the order and the energy of each diagram sampled during the MonteCarlo simulation.

The Polaron class is provided with some methods to evaluate the probability of accepting or rejecting an update and actually accepting or rejecting it based on the outcome of Metropolis-Hastings criterion. The two main functions are `eval_add_internal` and `eval_remove_internal` which always call some other class methods involved in the evaluation of the acceptance probability and if the update is accepted calls the `add_internal` and `remove_internal` method that either add or remove the specified phonon and updates the order of the current diagram. Another important method is `update_diagrams_info` which stores after each MonteCarlo steps the order and energy of the current diagram.

`dmc.py`
It contains the functions `run_thermalization_steps` and `run_diagrammatic_montecarlo` that run respectively some steps to thermalize the Markov chain and move statistically the system to a configuration with an higher probability and the actual Diagrammatic MonteCarlo simulation that performs the number of steps specified in _configuration.txt_.

`plot.py`
The main function is `plot_montecarlo` which:
- plot the histogram for the phonon distribution 
- fit the sampled histogram with the theoretical Poisson distribution to compare DiagMC results with the theory
- evaluate $E_{GS}$ and $N_{pho}$ and show in the legend
- save an image of the phonon distribution
- save the relevant parameters in an output file

`main.py`
Performs all the required steps:
- get parameters from _configuration.txt_
- initialize a Polaron object with the relevant parameters
- set the seed for sampling of random number to ensure reproducibility
- perform the MonteCarlo simulation and eventually a thermalization procedure
- store the relevant lists for statistical quantities and plot the histogram 

### Tests

To test the program in your environment execute
```pytest .```
it will automatically test all the functions whose name start with _test_. 
1. *test_parser.py* tests the behaviour of the custom parser `Config` and of the helper functions defined in `config_parser.py`.
2. *test_polaron.py* tests all the functions involved in the process of evaluating and eventually performing one of the two updates `add_internal` and `remove_internal` taking into account the possible outcomes of Metropolis-Hastings criterion based on the value of the acceptance probability for the chosen update.
3. *test_dmc.py* tests that `run_thermalization_steps` and `run_diagrammatic_montecarlo` performs exactly the number of steps specified in _configuration.txt_ and behaves as expected depending on the conditional statements 
