# File: neuron_model.py

import numpy as np

class Synapse:
    def __init__(self, g_max, E_syn, tau_syn):
        """
        Initialize synapse with maximum conductance, synaptic reversal potential, and time constant.
        
        :param g_max: Maximum synaptic conductance
        :param E_syn: Synaptic reversal potential
        :param tau_syn: Time constant for synaptic conductance
        """
        self.g_max = g_max
        self.E_syn = E_syn
        self.tau_syn = tau_syn
        self.g_syn = 0  # Current conductance
        self.s = 0  # State variable for filtering the synaptic input

    def update(self, dt, pre_spike_train):
        """
        Update synaptic conductance based on pre-synaptic spikes and decay over time.
        
        :param dt: Time step
        :param pre_spike_train: Boolean representing if there was a spike from the pre-synaptic neuron
        """
        # Update synaptic state based on whether a spike occurred and decay using the time constant
        self.s += pre_spike_train * self.g_max
        self.s *= np.exp(-dt / self.tau_syn)
        self.g_syn = self.s

    def get_conductance(self):
        """
        Return the current synaptic conductance.
        """
        return self.g_syn


class Neuron:
    def __init__(self, neuron_type="excitatory", V_rest=-70.0, V_threshold=-50.0, V_reset=-65.0, tau_membrane=20.0):
        """
        Initialize the neuron with its parameters.
        
        :param neuron_type: 'excitatory' or 'inhibitory' neuron
        :param V_rest: Resting membrane potential
        :param V_threshold: Threshold potential for spiking
        :param V_reset: Reset potential after spiking
        :param tau_membrane: Time constant for membrane voltage decay
        """
        self.neuron_type = neuron_type
        self.V = V_rest  # Membrane potential
        self.V_rest = V_rest  # Resting membrane potential
        self.V_threshold = V_threshold  # Spike threshold
        self.V_reset = V_reset  # Reset potential after spike
        self.tau_membrane = tau_membrane  # Membrane time constant
        self.spike_occurred = False

        # List to store synaptic inputs
        self.synapses = []

    def add_synapse(self, synapse):
        """
        Add a synapse to the neuron.
        
        :param synapse: Synapse object
        """
        self.synapses.append(synapse)

    def update_membrane_potential(self, dt):
        """
        Update the membrane potential based on synaptic inputs.
        
        :param dt: Time step for the update
        """
        total_synaptic_current = 0
        for synapse in self.synapses:
            g_syn = synapse.get_conductance()
            total_synaptic_current += g_syn * (self.V - synapse.E_syn)

        # Update the membrane potential using synaptic current and membrane time constant
        dV = (self.V_rest - self.V + total_synaptic_current) / self.tau_membrane
        self.V += dV * dt

    def apply_synaptic_input(self, dt, pre_spike_train):
        """
        Apply synaptic inputs from pre-synaptic neurons.
        
        :param dt: Time step for synaptic update
        :param pre_spike_train: List of pre-synaptic spike events (boolean)
        """
        for i, synapse in enumerate(self.synapses):
            synapse.update(dt, pre_spike_train[i])

    def check_for_spike(self):
        """
        Check if the neuron has spiked and reset the membrane potential if it has.
        """
        if self.V >= self.V_threshold:
            self.spike_occurred = True
            self.V = self.V_reset
        else:
            self.spike_occurred = False

    def step(self, dt, pre_spike_train):
        """
        Perform a single update step for the neuron.
        
        :param dt: Time step
        :param pre_spike_train: List of pre-synaptic spike events (boolean)
        """
        self.apply_synaptic_input(dt, pre_spike_train)
        self.update_membrane_potential(dt)
        self.check_for_spike()

# This file contains the base neuron and synapse model for integrate-and-fire neurons,
# handling synaptic inputs and membrane potential dynamics.

