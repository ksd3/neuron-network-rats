# File: simulation.py

import numpy as np

class Simulator:
    def __init__(self, network, time_window=1000, dt=1):
        """
        Initialize the simulator with the network and time window for the simulation.
        
        :param network: Network object containing neurons and their connections.
        :param time_window: Total time window for the simulation in ms.
        :param dt: Time step for the simulation in ms.
        """
        self.network = network
        self.time_window = time_window
        self.dt = dt
        self.time_steps = int(time_window / dt)
        self.spikes = []
        self.sub_threshold_voltages = []

    def simulate(self, pre_spike_trains):
        """
        Run the simulation for the given time window with pre-generated spike trains.
        
        :param pre_spike_trains: Pre-synaptic spike trains generated outside this function.
        """
        for t in range(self.time_steps):
            # Advance the network by one time step
            self.step(self.dt, pre_spike_trains, t)
            # Record spikes and sub-threshold membrane potentials
            self.record_sub_threshold_activity()
            self.record_spikes()

    def step(self, dt, pre_spike_trains, time_step):
        """
        Perform a single time step update for the entire network.
        
        :param dt: Time step for the update.
        :param pre_spike_trains: List of pre-synaptic spike events (boolean).
        :param time_step: The current time step in the simulation.
        """
        self.network.step(dt, pre_spike_trains, time_step)

    def record_sub_threshold_activity(self):
        """
        Record the sub-threshold membrane potential (below threshold) for all neurons.
        """
        # Detach the PyTorch tensors from the computation graph before converting to NumPy
        voltages = [neuron.V.detach().numpy() for neuron in self.network.neurons if not neuron.spike_occurred]
        if voltages:
            self.sub_threshold_voltages.append(np.mean(voltages))

    def record_spikes(self):
        """
        Record spikes for all neurons at the current time step.
        """
        spikes = [neuron.spike_occurred for neuron in self.network.neurons]
        self.spikes.append(spikes)

    def calculate_averaged_voltage(self):
        """
        Calculate the time-averaged membrane potential across the population.
        
        :return: Time-averaged membrane potential.
        """
        if self.sub_threshold_voltages:
            return np.mean(self.sub_threshold_voltages)
        return None
