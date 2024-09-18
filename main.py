# File: main.py

from tqdm import tqdm
from train_model import PyTorchNetwork, TrainModel
from network_model import Network  # To generate spike trains
from simulation import Simulator
from decision_making_task import DecisionTask
from evaluate_performance import PerformanceEvaluator
import torch

def main():
    # Step 1: Initialize the network using PyTorch
    num_neurons = 100  # Example number of neurons
    pytorch_network = PyTorchNetwork(num_neurons=num_neurons)  # Use PyTorch network

    # Step 2: Initialize the non-PyTorch Network for spike generation
    non_pytorch_network = Network(num_neurons=num_neurons)
    non_pytorch_network.connect_neurons()  # Connect neurons to use their spike trains

    # Step 3: Generate spike trains for the network
    time_window = 500  # Time in ms
    dt = 1  # Time step in ms
    print("Generating spike trains...")
    pre_spike_trains = non_pytorch_network.generate_spike_trains(rate=5, time_window=time_window, dt=dt)

    # Step 4: Simulate the PyTorch-based network
    simulator = Simulator(pytorch_network, time_window, dt)

    print("Simulating the network...")
    for time_step in tqdm(range(simulator.time_steps), desc="Simulation Progress"):
        simulator.step(dt, pre_spike_trains, time_step)  # Pass the current time_step to the step function

    # Step 5: Run decision-making task
    selective_neurons = pytorch_network.neurons[:10]  # Example selection of neurons for task condition
    task_condition = True  # Task condition example (can be more elaborate)
    decision_task = DecisionTask(pytorch_network, time_window, dt, task_duration=500)
    task_success = decision_task.run_trial(selective_neurons, task_condition)
    print(f"Task success: {task_success}")

    # Step 6: Train the PyTorch network
    train_model = TrainModel(pytorch_network)  # Use PyTorch network for training
    synaptic_inputs = torch.randn(num_neurons)  # Example synaptic input
    target_activity = torch.randn(num_neurons)  # Example target activity for training

    print("Training the network...")
    # Adding progress bar for the training loop
    for epoch in tqdm(range(10), desc="Training Progress"):  # Example training loop for 10 epochs
        loss = train_model.train_step(synaptic_inputs, target_activity)
        print(f"Epoch {epoch+1}, Loss: {loss}")

    # Step 7: Evaluate the network's performance
    evaluator = PerformanceEvaluator(simulator)

    # Example of experimental data (in practice, this would come from real data)
    experimental_spike_rates = torch.randn(num_neurons)
    experimental_isi = [torch.randn(10) for _ in range(num_neurons)]  # Example ISI data
    experimental_voltages = torch.randn(100)  # Example sub-threshold voltage data

    performance_metrics = evaluator.compare_with_experimental_data(
        experimental_spike_rates,
        experimental_isi,
        experimental_voltages
    )

    print("Performance Metrics:")
    print(f"Spike Rate Error: {performance_metrics['spike_rate_error']}")
    print(f"ISI Error: {performance_metrics['isi_error']}")
    print(f"Voltage Error: {performance_metrics['voltage_error']}")

if __name__ == "__main__":
    main()
