# File: network_model.py

import numpy as np
from dendritic_neuron_model import DendriticNeuron, Synapse

class Network:
    def __init__(self, num_neurons, excitatory_ratio=0.8):
        """
        Initialize the network with a given number of neurons.
        
        :param num_neurons: Total number of neurons in the network.
        :param excitatory_ratio: The proportion of neurons that are excitatory.
        """
        self.num_neurons = num_neurons
        self.excitatory_ratio = excitatory_ratio
        self.neurons = []

    def add_neuron(self, neuron_type="excitatory"):
        """
        Add a neuron to the network.
        
        :param neuron_type: Type of neuron, either 'excitatory' or 'inhibitory'.
        """
        neuron = DendriticNeuron(neuron_type)
        self.neurons.append(neuron)

    def connect_neurons(self, mean_connections=10, connection_std=2.0):
        """
        Connect neurons using a sparse log-normal distribution for the number of synapses per neuron.
        
        :param mean_connections: Mean number of connections per neuron.
        :param connection_std: Standard deviation for the log-normal distribution.
        """
        for neuron in self.neurons:
            num_connections = int(np.random.lognormal(mean_connections, connection_std))
            pre_synaptic_neurons = np.random.choice(self.neurons, num_connections, replace=False)
            for pre_neuron in pre_synaptic_neurons:
                # Assign different synapse parameters for excitatory and inhibitory neurons
                if pre_neuron.neuron_type == "excitatory":
                    g_max = 0.05  # example value for excitatory synapse strength
                    E_syn = 0.0   # excitatory synapse reversal potential
                else:
                    g_max = 0.1   # example value for inhibitory synapse strength
                    E_syn = -80.0  # inhibitory synapse reversal potential

                # Create synapse and add it to the current neuron
                synapse = Synapse(g_max=g_max, E_syn=E_syn, tau_syn=5.0)
                neuron.add_synapse(synapse)

    def generate_spike_trains(self, rate=5, time_window=1000, dt=1):
        """
        Generate random spike trains for each neuron.
        
        :param rate: The mean spike rate in Hz.
        :param time_window: The time window for the spike train in ms.
        :param dt: Time step in ms.
        :return: List of spike trains for each neuron in the network.
        """
        spike_trains = []
        for _ in range(self.num_neurons):
            # Generate Poisson distributed spikes for each neuron
            spikes = np.random.poisson(rate * dt / 1000, int(time_window / dt))
            spike_trains.append(spikes)
        return spike_trains

    def step(self, dt, pre_spike_trains):
        """
        Perform a single time step update for the entire network.
        
        :param dt: Time step for the update.
        :param pre_spike_trains: List of pre-synaptic spike events (boolean).
        """
        for neuron, pre_spike_train in zip(self.neurons, pre_spike_trains):
            neuron.step(dt, pre_spike_train)

# This file models a sparse network of neurons with excitatory and inhibitory synapses.
# Neurons are connected based on a log-normal distribution, and pre-synaptic spike trains
# influence post-synaptic potentials via conductance-based synaptic interactions.
