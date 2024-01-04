# Holstein Polaron Diagrammatic MonteCarlo

This program finds the ground state energy $E_{GS}$ of the Holstein hamiltonian
$$ H = \varepsilon\hat{c}^{\dag}\hat{c} + \omega\hat{b}^{\dag}\hat{b} + g(\hat{b}^{\dag} + \hat{b})\,\hat{c}^{\dag}\hat{c} $$
which describes a single electron with energy $\varepsilon$ in an atomic orbital  that couples via a vertex interaction of intensity $g$ to phonons of the lattice with frequency $\omega$.
This hamiltonian has an analytic solution both for $E_{GS}$, the mean number of phonons $N_{pho}$ and the phonon distribution $P(n)$
$$E_{GS} = -\frac{g^{2}}{\omega}$$
$$N_{pho} = \frac{g^{2}}{\omega^{2}} \quad \textrm{and} \quad P(n) = e^{N_{pho}}\,\frac{1}{n!}\,(N_{pho})^{n}$$

In this simulation we use a Markov Chain MonteCarlo algorithm to sample Feynman diagrams obtained in the series expansion of the Green's function. The simulation proceeds selecting randomly at each MonteCarlo step one of the possible updates for the current diagram which are _add\_phonon_ and _remove\_phonon_ and accept or reject it based on the outcome of Metropolis-Hastings criterion.

At each step we collect the order and evaluate the energy of the current diagram via the use of an estimator which averaged over the MonteCarlo samples retrieves $E_{GS}$
$$ \left\langle \frac{\sum_{i}^{pho} \omega \Delta\tau_{i} - \textrm{order}}{\tau}\right\rangle_{\textrm{DMC}} $$ 
the label _order_ in the program refers to the number of e-ph vertexes in the diagram instead of the actual order of the diagram in perturbation theory. 

Moreover, since we consider only diagrams where the electron is coupled to _internal_ phonons i.e. curly interactions lines whose both extrema have an intersection with the line of the electron it means that $order = 2* #phonons$.

For this reason collecting the order of the sampled diagrams let us evaluate $N_{pho}$ and $P(n)$ and plot the histogram for $P(n)$ just after dividing by 2 each occurrence for the order of a diagram. 
