# File: dendritic_neuron_model.py

from neuron_model import Neuron, Synapse

class DendriticSegment:
    def __init__(self, distance_from_soma, visibility=1.0, attenuation=1.0):
        """
        Represents a dendritic segment of a neuron where synaptic inputs are received.
        
        :param distance_from_soma: The distance of the dendritic segment from the soma.
        :param visibility: Visibility parameter Γ, controls the decay of synaptic influence with distance.
        :param attenuation: Attenuation parameter α, scales the synaptic input based on distance.
        """
        self.distance_from_soma = distance_from_soma
        self.visibility = visibility
        self.attenuation = attenuation
        self.synapses = []

    def add_synapse(self, synapse):
        """
        Add a synapse to the dendritic segment.
        
        :param synapse: Synapse object to be added to the dendrite.
        """
        self.synapses.append(synapse)

    def get_scaled_conductance(self):
        """
        Calculate the scaled conductance for the dendritic segment.
        The scaling depends on the attenuation and visibility parameters.
        """
        total_conductance = 0
        for synapse in self.synapses:
            total_conductance += synapse.get_conductance() * self.visibility * self.attenuation
        return total_conductance


class DendriticNeuron(Neuron):
    def __init__(self, neuron_type="excitatory", V_rest=-70.0, V_threshold=-50.0, V_reset=-65.0, tau_membrane=20.0):
        """
        Initialize the dendritic neuron with basic neuron parameters and an empty list of dendritic segments.
        """
        super().__init__(neuron_type, V_rest, V_threshold, V_reset, tau_membrane)
        self.dendrites = []

    def add_dendrite(self, distance_from_soma, visibility=1.0, attenuation=1.0):
        """
        Add a dendritic segment to the neuron.
        
        :param distance_from_soma: The distance of the dendritic segment from the soma.
        :param visibility: The visibility parameter Γ.
        :param attenuation: The attenuation parameter α.
        """
        dendrite = DendriticSegment(distance_from_soma, visibility, attenuation)
        self.dendrites.append(dendrite)
        return dendrite

    def apply_dendritic_input(self, dt, pre_spike_train):
        """
        Apply synaptic input to all dendritic segments, taking into account the distance from soma.
        
        :param dt: Time step for synaptic update.
        :param pre_spike_train: List of pre-synaptic spike events (boolean).
        """
        for dendrite in self.dendrites:
            for i, synapse in enumerate(dendrite.synapses):
                synapse.update(dt, pre_spike_train[i])

    def update_membrane_potential(self, dt):
        """
        Update the membrane potential, considering both somatic and dendritic synaptic inputs.
        
        :param dt: Time step for the update.
        """
        total_synaptic_current = 0

        # Calculate total current from all dendritic segments
        for dendrite in self.dendrites:
            scaled_conductance = dendrite.get_scaled_conductance()
            for synapse in dendrite.synapses:
                total_synaptic_current += scaled_conductance * (self.V - synapse.E_syn)

        # Update the membrane potential using synaptic current and membrane time constant
        dV = (self.V_rest - self.V + total_synaptic_current) / self.tau_membrane
        self.V += dV * dt

    def step(self, dt, pre_spike_train):
        """
        Perform a single update step for the neuron, applying dendritic inputs and updating the membrane potential.
        
        :param dt: Time step for the update.
        :param pre_spike_train: List of pre-synaptic spike events (boolean).
        """
        self.apply_dendritic_input(dt, pre_spike_train)
        self.update_membrane_potential(dt)
        self.check_for_spike()

# This file extends the neuron model to include dendritic morphology. Dendritic segments
# influence the neuron's somatic membrane potential based on their distance and the associated
# visibility (Γ) and attenuation (α) parameters.

