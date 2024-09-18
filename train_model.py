import torch
import torch.nn as nn
import torch.optim as optim

class PyTorchNeuron(nn.Module):
    def __init__(self, V_rest=-70, V_threshold=-50, V_reset=-65, tau_membrane=20):
        super(PyTorchNeuron, self).__init__()
        self.V_rest = torch.nn.Parameter(torch.tensor(V_rest, dtype=torch.float32), requires_grad=True)
        self.V_threshold = torch.nn.Parameter(torch.tensor(V_threshold, dtype=torch.float32), requires_grad=True)
        self.V_reset = torch.nn.Parameter(torch.tensor(V_reset, dtype=torch.float32), requires_grad=True)
        self.tau_membrane = torch.nn.Parameter(torch.tensor(tau_membrane, dtype=torch.float32), requires_grad=True)

        # Membrane potential (not a parameter, no gradient tracking)
        self.V = torch.tensor([V_rest], requires_grad=False)

        # Initialize spike_occurred attribute
        self.spike_occurred = False

    def forward(self, synaptic_input):
        dV = (self.V_rest - self.V + synaptic_input) / self.tau_membrane

        # Update the membrane potential without tracking gradients
        with torch.no_grad():
            self.V = self.V + dV

        # Check for spike occurrence
        if self.V >= self.V_threshold:
            self.spike_occurred = True
            with torch.no_grad():
                self.V = self.V_reset
        else:
            self.spike_occurred = False

        return self.V, self.spike_occurred





class PyTorchNetwork(nn.Module):
    def __init__(self, num_neurons, excitatory_ratio=0.8):
        super(PyTorchNetwork, self).__init__()
        self.num_neurons = num_neurons
        self.excitatory_ratio = excitatory_ratio

        self.neurons = nn.ModuleList([PyTorchNeuron() for _ in range(num_neurons)])

    def forward(self, synaptic_inputs):
        potentials = []
        spikes = []

        for i, neuron in enumerate(self.neurons):
            neuron_input = synaptic_inputs[i]
            V, spike = neuron(neuron_input)
            potentials.append(V)
            # Here we need to make sure the spikes are tensors that track gradients
            spikes.append(torch.tensor(float(spike), dtype=torch.float32))  # Ensure spikes are tensors

        # Stack spikes into a tensor and make sure they require gradients
        spikes_tensor = torch.stack(spikes).requires_grad_(True)
        return potentials, spikes_tensor


    def step(self, dt, pre_spike_trains, time_step):
        """
        Perform a single time step update for the PyTorch-based network.
        
        :param dt: Time step for the update.
        :param pre_spike_trains: List of pre-synaptic spike events for all neurons.
        :param time_step: Current time step in the simulation.
        """
        current_inputs = torch.tensor([spike_train[time_step] for spike_train in pre_spike_trains], dtype=torch.float32)
        return self.forward(current_inputs)


class TrainModel:
    def __init__(self, network, learning_rate=0.001):
        """
        Initialize the training process for the neuron network.
        
        :param network: The PyTorch-based neuron network.
        :param learning_rate: Learning rate for the optimizer.
        """
        self.network = network
        self.optimizer = optim.Adam(self.network.parameters(), lr=learning_rate)
        self.loss_fn = nn.MSELoss()  # Loss function, for example, Mean Squared Error

    def train_step(self, synaptic_inputs, target_activity):
        """
        Perform a single training step.
        
        :param synaptic_inputs: Synaptic inputs for the neurons.
        :param target_activity: Target activity (e.g., spike rates) to match.
        """
        self.optimizer.zero_grad()

        # Forward pass through the network
        potentials, spikes = self.network(synaptic_inputs)

        # Compute loss based on the target activity
        loss = self.loss_fn(spikes, target_activity)

        # Backpropagation and optimization step
        loss.backward()
        self.optimizer.step()

        return loss.item()

# This file provides a PyTorch-based implementation of the neuron network model.
# It includes a training process to optimize synaptic weights and other parameters
# using gradient-based optimization and backpropagation.
