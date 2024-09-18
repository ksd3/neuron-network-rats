# File: evaluate_performance.py

import numpy as np
import torch

class PerformanceEvaluator:
    def __init__(self, simulator):
        """
        Initialize the performance evaluator with a simulator object that contains the network's activity.
        
        :param simulator: Simulator object that has recorded spiking and voltage data.
        """
        self.simulator = simulator

    def compute_spike_rates(self):
        """
        Compute the spike rates of each neuron in the network.
        
        :return: List of spike rates for all neurons.
        """
        num_spikes = np.sum(self.simulator.spikes, axis=0)  # Total spikes per neuron
        time_window = self.simulator.time_window / 1000  # Convert ms to seconds
        spike_rates = num_spikes / time_window  # Spikes per second (Hz)
        return spike_rates

    def compute_isi_distribution(self):
        """
        Compute the inter-spike interval (ISI) distribution for all neurons.
        
        :return: List of ISIs for all neurons.
        """
        isi_list = []
        for neuron_spikes in self.simulator.spikes:
            spike_times = np.where(neuron_spikes == 1)[0]  # Find time indices where spikes occurred
            if len(spike_times) > 1:
                isis = np.diff(spike_times) * self.simulator.dt  # Compute ISI in ms
                isi_list.append(isis)
        return isi_list

    def compute_voltage_distribution(self):
        """
        Compute the distribution of sub-threshold membrane potentials across neurons.
        
        :return: List of sub-threshold membrane potentials.
        """
        return self.simulator.sub_threshold_voltages

    def compare_with_experimental_data(self, experimental_spike_rates, experimental_isi, experimental_voltages):
        """
        Compare the model's output with empirical data from cortex recordings.
        
        :param experimental_spike_rates: Experimental spike rate data.
        :param experimental_isi: Experimental inter-spike interval data.
        :param experimental_voltages: Experimental sub-threshold voltage data.
        :return: Dictionary with comparison metrics (e.g., error or correlation).
        """
        # Model data
        model_spike_rates = self.compute_spike_rates()
        model_isi = self.compute_isi_distribution()
        model_voltages = self.compute_voltage_distribution()

        # Convert PyTorch tensors to NumPy arrays if needed
        if isinstance(model_spike_rates, torch.Tensor):
            model_spike_rates = model_spike_rates.detach().numpy()
        if isinstance(experimental_spike_rates, torch.Tensor):
            experimental_spike_rates = experimental_spike_rates.detach().numpy()

        # Convert model_voltages (list) to NumPy array
        model_voltages = np.array(model_voltages)

        # Convert experimental_voltages to NumPy array if it's a tensor
        if isinstance(experimental_voltages, torch.Tensor):
            experimental_voltages = experimental_voltages.detach().numpy()

        # If model_voltages is empty, handle it
        if model_voltages.size == 0:
            voltage_error = float('nan')  # Set to NaN or some default value
        else:
            voltage_error = np.mean((model_voltages - experimental_voltages) ** 2)

        # Compute differences (e.g., mean squared error) between model and experimental data
        spike_rate_error = np.mean((model_spike_rates - experimental_spike_rates) ** 2)
        isi_error = np.mean([np.mean((model - exp) ** 2) for model, exp in zip(model_isi, experimental_isi)])

        return {
            "spike_rate_error": spike_rate_error,
            "isi_error": isi_error,
            "voltage_error": voltage_error
        }
