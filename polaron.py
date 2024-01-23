import numpy as np

class Polaron:
    """This class contains the properties of an Holstein polaron and the
    updates that are performed in the MonteCarlo simulation.

    Attributes
    ----------
    diagram :
        stores the general properties of a Feynman diagram
        produced during the simulation:
        - g : Intensity of electron-phonon coupling
        - omega : Frequency of the phonons that couple to the extra electron
        - time : Lifetime of the polaron
        - energy : Energy of the diagram evaluated through an estimator
        - order : Order of the current Feynman diagram sampled

    phonon_list :
        stores the generation and removal time of the phonons that
        couples to the electron. These are scaled imaginary time values.
    
    energy_sequence :
        energy values evaluated through an estimator for each of the
        sampled diagrams

    Methods
    -------
    metropolis(prob):
        Metropolis algorithm to accept or reject the proposed update

    add_phonon_scaling():
        factor to add when considering the ratio between the values of
        two Feynman diagrams due to the use of imaginary scaled times.

    weigth_ratio_add(phonon):
        ratio between the proposed Feynman diagram with an extra phonon
        and the current one

    proposal_add_ratio(phonon):
        ratio between the proposal probability for the inverse process
        where the current state is proposed from a state with an extra phonon
        and the proposal probality for a state with an extra phonon

    add_internal():
        add a phonon to the phonon_list and updates the order of the
        diagram

    eval_add_internal():
        evaluate the acceptance probability for the introduction of
        a phonon and eventually add it to the current diagram
        depending on the outcome of Metropolis-Hastings.

    choose_phonon() :
        retrieve randomly a phonon from the ones stored in the diagram

    weigth_ratio_remove(phonon) :
        ratio between the value of the proposed Feynman diagram without
        one the phonons present in the current diagram and the current one

    proposal_remove_ratio(phonon) :
        ratio between the proposal probality for the current state from the
        proposed state with one phonon removed and the proposal probality
        for a state with one less phonon

    remove_internal() :
        remove a phonon to the current diagram and update the order

    eval_remove_internal() :
        choose one of the phonons randomly and evaluate the
        acceptance probability for the removal of
        the phonon eventually removing it from the current diagram
        depending on the outcome of Metropolis-Hastings

    eval_diagram_energy():
        evaluate the energy contribution to the polaron energy from
        the current Feynman diagram

    set_starting_info():
        initialize the lists for the order, energy and lifetime
        sequences after the thermalization of the Markov chain

    """

    def __init__(self, omega : float, g : float, time : float):
        """This function builds:
            - diagram : a dictionary that stores the features of the Feynman's diagram of a polaron.
                It stores both fixed parameters and variables read from a configuration file.

            - phonon_list : empty list that is updated at each step to store the phonons 
                coupled to the electron at the current MonteCarlo step

            - order_sequence : empty list that will store the order of each diagram sampled during the
                MonteCarlo simulation. It will we used to reproduce the sampled distribution for the number 
                of phonons (n_phonons = order/2)
            
            - energy_sequence : empty list that will store the energy of each diagram sampled during the
                MonteCarlo simulation. It will we used find the ground state energy of the polaron
                using the energy estimator

        Parameters:
            omega : frequency of the phonons
            g : intensity of electron-phonon coupling
            time : lifetime of the electron
        """
        self.diagram = {}
        self.diagram['omega'] = omega
        self.diagram['g'] = g
        self.diagram['time'] = time
        self.diagram['order'] = 0
        self.diagram['total_energy'] = 0.0
      
        self.phonon_list = []
        self.order_sequence = []
        self.energy_sequence = []
 

    def metropolis(self, prob : float) -> int | float:
        """This function uses the Metropolis choice for the 
            detailed balance of the Markov chain
        
        Parameters:
            prob : ratio between the acceptance probability for an update
                and its reverse
                
        Returns:
            minimum between one and the ratio of acceptance probabilities for
            an update and its reverse"""
        
        return min(1, prob)

    def add_phonon_scaling(self) -> float :
        """This method evaluate the scaling factor for the value
         of each diagram due to the presence of a phonon coupled to the electron. 
        
        Return:
            electron-phonon coupling times electron lifetime to the squared.
        ---------------------
        Notes:
            this factor comes from the use of imaginary scaled times
        """
        
        return (self.diagram['g']*self.diagram['time'])**2


    def weigth_ratio_add(self, phonon : np.ndarray) -> float :
        """This method evaluate the ratio between the proposed diagram with an additional
        phonon and the current one.

        - The ratio corresponds to the contribution given by a phonon interaction line 
        in a diagram
        
        Parameters:
            phonon : array that contains the initial and final time for the interaction
                line of the proposed phonon
            
        Return:
            scaling factor due to imaginary scaled times multiplied by the contribution of 
            a phonon propagator (phonon interaction line)"""
        
        phonon_propagator = np.exp(-self.diagram['time']*self.diagram['omega'] *
                            (phonon[1] - phonon[0]))
        return self.add_phonon_scaling()*phonon_propagator

    def proposal_add_ratio(self, phonon : np.ndarray) -> float :
        """This method evaluate the ratio between the proposal probability of choosing the reverse
        update (i.e. removing the phonon added) and the current one (i.e. adding the phonon)

        - reverse update: probability to pick up a specific phonon among many i.e. 1/(# of phonons)
        - direct update: probability to generate the picked phonon with those specific times i.e. 1*(1-t_gen)

        Parameters:
            phonon: array that contains the initial and final time for the interaction
                line of the proposed phonon

        Return:
            ratio between proposal probability for the reverse update and the direct one 
        -------------------------

        Notes:
        
        - Depending on the order of the diagram we have a prefactor from to the ratio
         between the probability to pick up the update for the reverse process (i.e. 
         remove the proposed phonon) and the probability to pick up the direct 
         process (i.e. add the desired phonon). 
         For an order 0 diagram the probability to pick up the remove update for the reverse process
         (from a diagram of order 2 with the proposed phonon) is 1/2 while the probability to pick up
         the add phonon update is 1 because at order 0 the remove update is meaningless

        - 1/(# of phonons) is 1/(len(self.phonon_list)+1) because it is the probability to 
         pick up the choosen phonon among the one coupled to the electron in the current diagram +
         the proposed one
        """
        if self.diagram['order'] == 0 :
            # direct update at order 0 can be only add_phonon 
            return 1/2*(1-phonon[0])/(len(self.phonon_list)+1)
        elif self.diagram['order'] != 0 :
            # order != 0 direct can be either add or remove phonon
            return (1-phonon[0])/(len(self.phonon_list)+1)
        
    def add_internal(self, phonon : np.ndarray):
        """This method append a phonon to the list of phonons already coupled
        to the electron and update the order of the diagram
        
        Parameters:
            phonon: array that contains the initial and final time for the interaction
                line of the proposed phonon
        """

        self.phonon_list.append(phonon)
        self.diagram['order'] += 2

    def eval_add_internal(self):
        """This method generates a phonon and evaluates the ratio
        between acceptance probabilities used in Metropolis-Hastings
        Depending on the outcome the proposed update can be accepted
        or rejected 

        Notes:
        If the update is accepted both the state of phonon_list and the
        order of the diagram change.
        One more phonon is appended and the order of the diagram increases.
        """

        # generate two random time for the extrema of the phonon interaction line
        t_gen = np.random.uniform(0, 1)
        t_rem = np.random.uniform(t_gen, 1)
        phonon = np.array([t_gen,t_rem])

        #evaluate the ratio between the acceptance probabilities of the current update and the reverse one
        ratio_acceptance_probs = self.weigth_ratio_add(phonon) * \
                self.proposal_add_ratio(phonon)
        acceptance = self.metropolis(ratio_acceptance_probs)

        #add phonon if metropolis returns 1
        if acceptance == 1:
            self.add_internal(phonon)
        
        #pick a random number between 0 and 1 and accept the update if the acceptance probability is greater
        elif 0 <= acceptance < 1:
            sample = np.random.uniform(0,1)
            if sample <= acceptance:
                self.add_internal(phonon)

    def choose_phonon(self) -> int :
        """This method selects randomly the index of a phonon among
        the indexes of the phonons in the list 

        Return:
            phonon_index: randomly chosen index of a phonon among
                the ones in the list
        """
        
        phonon_index = np.random.randint(len(self.phonon_list))
        return phonon_index

    def weigth_ratio_remove(self, phonon : np.ndarray) -> float:
        """This method evaluate the ratio between the proposed diagram with one less phonon
        phonon and the current one 

        - The ratio corresponds to the contribution given by the inverse of a 
        phonon interaction line in a diagram
        
        Parameters:
            phonon : array that contains the initial and final time for the interaction
                line of the selected phonon
            
        Return:
            Contribution from the reciprocal of a phonon propagator (phonon interaction line)
            divided by the scaling factor due to imaginary scaled times"""
        
        phonon_propagator_inverse = np.exp(self.diagram['time']*self.diagram['omega']*
                                (phonon[1] - phonon[0]))
        return phonon_propagator_inverse/self.add_phonon_scaling()

    def proposal_remove_ratio(self, phonon : np.ndarray) -> float :
        """This method evaluate the ratio between the proposal probability of choosing the reverse
        update (i.e. adding the phonon) and the current one (i.e. removing the chosen phonon)

        - reverse update: probability to generate the picked phonon with those specific times i.e. 1*(1-t_gen)
        - direct update: probability to pick up a specific phonon among many i.e. 1/(# of phonons)

        Parameters:
            phonon: array that contains the initial and final time for the interaction
                line of the proposed phonon

        Return:
            ratio between proposal probability for the reverse update and the direct one 
        -------------------------

        Notes:
        
        - Depending on the order of the diagram we have a prefactor from to the ratio
         between the probability to pick up the update for the reverse process (i.e. 
         add the chosen phonon) and the probability to pick up the direct 
         process (i.e. remove the chosen phonon). 
         For an order 2 diagram the probability to pick up the add update for the reverse process
         (from a diagram of order 0) is 1 while the probability to pick up
         the remove phonon update is 1/2. So the prefactor in this case is 2
        """

        if self.diagram['order'] == 2 :
            return 2*len(self.phonon_list)/(1-phonon[0])
        elif self.diagram['order'] != 2 :
            return len(self.phonon_list)/(1-phonon[0])

    def remove_internal(self, phonon_index : int):
        """This method remove a phonon from the list of phonons
        and update the order the diagram
        
        Parameters:
            phonon_index: index in phonon_list corresponding
                        to the chosen phonon
        Notes:
        This method changes the state of the object both decreasing 
        the order of the diagram and 
        """

        del self.phonon_list[phonon_index]
        self.diagram['order'] -= 2

    def eval_remove_internal(self):
        """This method evaluates the ratio between acceptance probabilities 
        used in Metropolis-Hastings for the remove_internal update.
        Depending on its outcome the update can be accepted or rejected.

        Notes:
        If the update is accepted both the state of phonon_list and the
        order of the diagram change.
        One phonon is removed from the list and the order of the diagram
        decreases.
        """

        #get a phonon randomly from the one coupled to the electron in the current diagram
        phonon_index = self.choose_phonon()
        phonon = self.phonon_list[phonon_index]

        #evaluate the ratio between the acceptance probabilities of the current update and the reverse one
        ratio_acceptance_probs = self.weigth_ratio_remove(phonon) * \
        self.proposal_remove_ratio(phonon)
        acceptance = self.metropolis(ratio_acceptance_probs)

        #remove phonon if metropolis returns 1
        if acceptance == 1:
            self.remove_internal(phonon_index)
        
        #pick a random number between 0 and 1 and accept the update if the acceptance probability is greater
        elif 0 <= acceptance < 1 :
            sample = np.random.uniform(0,1)
            if sample <= acceptance:
                self.remove_internal(phonon_index)

    def eval_diagram_energy(self):
        """This method evaluates the energy of the system at a MonteCarlo step
        using the formula of the estimator and updates the energy of the diagram
        """
        if self.diagram['order'] == 0:
            self.diagram['total_energy'] = 0.0
        else:
            phonons_gen_time_sum, phonons_rem_time_sum = np.sum(self.phonon_list, axis=0)
            phonons_interaction_time_sum = phonons_rem_time_sum - phonons_gen_time_sum
            self.diagram['total_energy'] = self.diagram['omega'] * phonons_interaction_time_sum - self.diagram['order']/ \
                                            self.diagram['time']

    def update_diagrams_info(self):
        """This method update the lists of diagrams order and
        energy of the polaron"""
        self.order_sequence.append(self.diagram['order'])
        self.energy_sequence.append(self.diagram['total_energy'])
