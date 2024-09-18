# File: decision_making_task.py

from simulation import Simulator
from network_model import Network  # Use the non-PyTorch Network for spike train generation

class DecisionTask:
    def __init__(self, network, time_window=1000, dt=1, task_duration=500):
        """
        Initialize the decision task with a given network and task parameters.
        
        :param network: The neural network used for the task (PyTorch-based network).
        :param time_window: Total time window for the simulation in ms.
        :param dt: Time step for the simulation in ms.
        :param task_duration: Duration of the decision-making task in ms.
        """
        self.network = network
        self.simulator = Simulator(network, time_window, dt)
        self.task_duration = task_duration

        # Initialize a non-PyTorch network for generating spike trains
        self.spike_train_generator = Network(len(network.neurons))

    def run_trial(self, selective_neurons, task_condition):
        """
        Run a single trial of the decision-making task.
        
        :param selective_neurons: List of neurons selective for the task condition.
        :param task_condition: Condition that biases certain neurons (e.g., stimulus or preparatory state).
        :return: Result of the trial (success or error).
        """
        # Simulate task-specific input to selective neurons
        for neuron in selective_neurons:
            if task_condition:
                neuron.V += 5  # Task bias to increase excitability in selective neurons

        # Generate spike trains using the non-PyTorch Network
        spike_trains = self.spike_train_generator.generate_spike_trains(rate=5, time_window=self.task_duration, dt=1)

        # Run the simulation for the duration of the task using the generated spike trains
        self.simulator.simulate(spike_trains)  # Pass spike_trains into simulate()

        # Evaluate neuron activity and determine task success
        return self.evaluate_performance(selective_neurons)

    def evaluate_performance(self, selective_neurons):
        """
        Evaluate task performance by checking selective neurons' responses.
        
        :param selective_neurons: List of neurons selective for the task condition.
        :return: Boolean indicating whether the trial was successful (True) or an error (False).
        """
        # Check if the selective neurons show spiking activity during the task
        spike_counts = [sum(self.simulator.spikes[i]) for i, neuron in enumerate(selective_neurons)]

        # Define success as at least some selective neurons spiking more than a threshold
        if any(spike_count > 3 for spike_count in spike_counts):
            return True  # Task success
        return False  # Task error
