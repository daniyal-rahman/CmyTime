import numpy as np
import pandas as pd
from src.brains.base_brain import BaseBrain
from src.models.lif_neuron import LIFNeuron # Import LIFNeuron

class LearningBrain(BaseBrain):
    def __init__(self, connectome_path, neuron_types_path, learning_rate=0.01, tau_pre=20, tau_post=20):
        self.connectome_path = connectome_path
        self.connectome, self.neuron_to_idx, self.all_neuron_names = self._load_connectome(connectome_path)
        self.neuron_types = self._categorize_neurons(neuron_types_path)
        
        self.sensory_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['sensory'] if n in self.neuron_to_idx]
        self.motor_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['motor'] if n in self.neuron_to_idx]
        
        # Initialize LIF neurons
        self.neurons = [LIFNeuron() for _ in range(len(self.all_neuron_names))]

        # Learning parameters
        self.learning_rate = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post

        # Neuron state variables for STDP
        num_neurons = len(self.all_neuron_names)
        self.pre_synaptic_trace = np.zeros(num_neurons)
        self.post_synaptic_trace = np.zeros(num_neurons)
        self.eligibility_trace = np.zeros_like(self.connectome)

        # Keep track of current time for LIF neuron stepping
        self.current_time = 0.0

        print(f"Initialized LearningBrain with STDP and LIF neurons: {len(self.sensory_neurons_idx)} sensory, {len(self.motor_neurons_idx)} motor neurons")

    def _load_connectome(self, path):
        df = pd.read_excel(path, sheet_name='Sheet1')
        sender_neurons = set(df['Neuron 1'].tolist())
        receiver_neurons = set(df['Neuron 2'].tolist())
        all_neurons = sorted(list(sender_neurons.union(receiver_neurons)))
        neuron_to_idx = {name: i for i, name in enumerate(all_neurons)}
        adj_matrix = np.zeros((len(all_neurons), len(all_neurons)))
        for _, row in df.iterrows():
            i = neuron_to_idx[row['Neuron 1']]
            j = neuron_to_idx[row['Neuron 2']]
            adj_matrix[i, j] = row['Nbr']
        return adj_matrix, neuron_to_idx, all_neurons

    def _categorize_neurons(self, path):
        xls = pd.ExcelFile(path)
        if 'NeuronsToMuscle' in xls.sheet_names:
            df = pd.read_excel(path, sheet_name='NeuronsToMuscle')
            if 'Function' in df.columns:
                sensory_neurons = df[df['Function'].str.contains('sensory', case=False, na=False)]['Neuron'].tolist()
                motor_neurons = df[df['Function'].str.contains('motor', case=False, na=False)]['Neuron'].tolist()
                if sensory_neurons or motor_neurons:
                    return {'sensory': sensory_neurons, 'motor': motor_neurons}
        df = pd.read_excel(self.connectome_path, sheet_name='Sheet1')
        all_neurons = pd.concat([df['Neuron 1'], df['Neuron 2']]).unique()
        sensory_neurons = [n for n in all_neurons if 'S' in n]
        motor_neurons = [
            'VB1', 'VB2', 'VB3', 'VB4', 'VB5', 'VB6', 'VB7', 'VB8', 'VB9', 'VB10', 'VB11',
            'DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB6', 'DB7',
            'VA1', 'VA2', 'VA3', 'VA4', 'VA5', 'VA6', 'VA7', 'VA8', 'VA9', 'VA10', 'VA11', 'VA12',
            'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6', 'DA7', 'DA8', 'DA9',
            'VD1', 'VD2', 'VD3', 'VD4', 'VD5', 'VD6', 'VD7', 'VD8', 'VD9', 'VD10', 'VD11', 'VD12', 'VD13',
            'VC1', 'VC2', 'VC3', 'VC4', 'VC5', 'VC6',
            'AS1', 'AS2', 'AS3', 'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AS10', 'AS11'
        ]
        return {'sensory': sensory_neurons, 'motor': motor_neurons}

    def get_action(self, observation):
        # Increment current time for LIF neuron stepping
        self.current_time += 1.0

        # --- Calculate Input Current for each Neuron ---
        # Initialize input current for all neurons
        input_current = np.zeros(len(self.all_neuron_names))

        # Apply sensory input for food
        # Increased scaling for food sensory input
        food_sensory_input_scaled_x = observation['gradient_x'] * 10.0 # Increased scaling
        food_sensory_input_scaled_y = observation['gradient_y'] * 10.0 # Increased scaling
        input_current[self.sensory_neurons_idx[0]] += food_sensory_input_scaled_x
        input_current[self.sensory_neurons_idx[1]] += food_sensory_input_scaled_y

        # Apply sensory input for traps
        if 'distance_to_nearest_trap' in observation:
            trap_gradient = -1 / (1 + observation['distance_to_nearest_trap'])
            # Increased scaling for trap sensory input
            trap_sensory_input_scaled_x = trap_gradient * np.cos(observation['relative_trap_angle']) * 10.0 # Increased scaling
            trap_sensory_input_scaled_y = trap_gradient * np.sin(observation['relative_trap_angle']) * 10.0 # Increased scaling
            input_current[self.sensory_neurons_idx[2]] += trap_sensory_input_scaled_x
            input_current[self.sensory_neurons_idx[3]] += trap_sensory_input_scaled_y

        # Propagate activation through the connectome to all neurons
        # Each neuron receives input from all other neurons connected to it
        # The input current for a neuron is the sum of (weight * spike_output_from_pre_synaptic_neuron)
        # For simplicity, we'll use the membrane potential of the previous step as a proxy for "spike_output"
        # This is a simplification; ideally, this would be based on actual spikes from the previous step.
        # For now, let's use the current membrane potentials as a base for input to the next step.
        # This requires a slight conceptual shift: the input_current here is what *drives* the LIF neurons.
        # The actual "activation" of neurons comes from their internal LIF step.

        # Calculate input from other neurons based on current membrane potentials and connectome
        # This is a simplified way to get recurrent input. A more accurate model would use actual spikes.
        # For now, let's assume the input current is directly influenced by the weighted sum of previous membrane potentials.
        # This is a placeholder and might need refinement.
        
        # Let's refine this: the input current to a neuron should be the sum of weighted spikes from presynaptic neurons.
        # Since LIFNeuron.step returns 1.0 for a spike and 0.0 otherwise, we need to get the spikes from the previous step.
        # However, get_action is called once per simulation step.
        # So, the input current for the *current* step should be based on the *previous* step's spikes.
        # We need to store the spikes from the previous step.

        # For now, let's use the current sensory input and then propagate through the connectome.
        # The `input_current` array will be the direct input to the LIF neurons.
        # The `connectome.T @ input_current` was conceptually wrong for LIF.
        # The input to LIF neurons should be the sum of weighted spikes from *presynaptic* neurons.

        # Let's assume for now that the `input_current` array is the external input to the sensory neurons,
        # and then the internal dynamics of the LIF neurons and the connectome will handle the rest.

        # The `input_current` array is what goes into the `LIFNeuron.step` method.
        # So, we need to calculate the total input for *each* neuron.
        # This total input comes from sensory input AND from other neurons via the connectome.

        # Let's calculate the input from other neurons based on their *current* membrane potentials (as a proxy for activity)
        # This is a simplification. A more rigorous approach would involve tracking spikes over time.
        # For now, let's use the membrane potential of the previous step as the "output" of a neuron.
        
        # This is the core change: how do we get the input to each LIF neuron?
        # It should be the sum of weighted outputs (spikes) from its presynaptic neurons.
        # Since we are stepping all neurons at once, we need to consider the "state" from the previous time step.

        # Let's assume `self.neuron_outputs` stores the spike outputs from the *previous* step.
        # Initialize `self.neuron_outputs` in `__init__` to zeros.
        # Then, in `get_action`, calculate `input_current` for each neuron.

        # Input from other neurons (recurrent connections)
        # This assumes `self.neuron_outputs` holds the spike outputs from the *previous* time step.
        # We need to ensure `self.neuron_outputs` is updated after all neurons have stepped.
        
        # For the first pass, let's simplify: the input to each neuron is the sum of sensory input
        # plus the weighted sum of the *current* membrane potentials of its presynaptic neurons.
        # This is not ideal for true spiking dynamics but gets us closer to a working model.

        # Let's use the current membrane potentials as the "output" for calculating recurrent input.
        # This is a common simplification in rate-based models, but we are using LIF.
        # The correct way is to use the *spikes* from the previous time step.
        # Let's store the spikes from the previous step.

        previous_spikes = np.array([neuron.spike_output for neuron in self.neurons]) # Assuming spike_output is stored

        # Add recurrent input from other neurons based on previous spikes
        # The input to neuron `j` is sum(weight_ij * spike_i) for all `i`
        # So, `connectome.T @ previous_spikes` gives the input to each neuron.
        input_current += self.connectome.T @ previous_spikes

        # --- Neuron Stepping and Spiking ---
        current_spikes = np.zeros(len(self.all_neuron_names))
        for i, neuron in enumerate(self.neurons):
            current_spikes[i] = neuron.step(input_current[i], self.current_time)
        
        # Store current spikes for next iteration's recurrent input calculation
        for i, neuron in enumerate(self.neurons):
            neuron.spike_output = current_spikes[i] # Assuming LIFNeuron has a spike_output attribute

        # --- STDP Trace Updates (based on actual LIF neuron spikes) ---
        self.pre_synaptic_trace *= np.exp(-1.0 / self.tau_pre)
        self.post_synaptic_trace *= np.exp(-1.0 / self.tau_post)
        self.pre_synaptic_trace[current_spikes == 1.0] = 1.0
        self.post_synaptic_trace[current_spikes == 1.0] = 1.0
        
        # Eligibility trace update based on actual spikes
        # This needs to be carefully considered for a sparse connectome.
        # For now, let's use the outer product, which works for dense matrices.
        # If connectome is sparse, this might be inefficient.
        self.eligibility_trace += np.outer(self.pre_synaptic_trace, current_spikes)
        self.eligibility_trace -= np.outer(current_spikes, self.post_synaptic_trace)

        # --- Action Generation (based on motor neuron activity) ---
        # Use the membrane potential of motor neurons to determine action
        motor_neuron_potentials = np.array([self.neurons[idx].membrane_potential for idx in self.motor_neurons_idx])
        
        # Normalize motor potentials to get a signal between 0 and 1
        # Avoid division by zero if all potentials are the same (e.g., all zero)
        if motor_neuron_potentials.max() - motor_neuron_potentials.min() > 1e-6:
            normalized_motor_signal = (motor_neuron_potentials - motor_neuron_potentials.min()) / \
                                      (motor_neuron_potentials.max() - motor_neuron_potentials.min())
        else:
            normalized_motor_signal = np.zeros_like(motor_neuron_potentials)

        # Average the normalized signals to get a base forward signal
        base_forward_signal = np.mean(normalized_motor_signal) if len(normalized_motor_signal) > 0 else 0

        # Action Generation (based on motor neuron activity and environmental cues)
        # Combine a direct influence from relative stimulus angle with motor neuron activity for turning.
        # This provides a stronger initial signal for food-seeking that STDP can refine.
        relative_angle = observation['relative_stimulus_angle'][0]
        distance_to_stimulus = observation['distance_to_stimulus'][0]
        
        # Reduce turning influence when very close to stimulus to prevent circling
        # Linear decay with a very small minimum, allowing for fine adjustments
        turn_strength_factor = np.clip(distance_to_stimulus / 15.0, 0.01, 1.0) # Scale factor, min 0.01 for very fine adjustments
        turn_action = (relative_angle / np.pi) * turn_strength_factor # Direct influence from stimulus angle, scaled

        # Incorporate trap avoidance directly into turn_action
        if 'distance_to_nearest_trap' in observation and observation['distance_to_nearest_trap'] < 30: # Further increased trap detection radius
            # Prioritize trap avoidance when very close to a trap
            turn_action = -observation['relative_trap_angle'][0] / np.pi * 8.0 # Even stronger turning away from trap
            move_forward_signal = 0.0 # Halt forward movement when a trap is detected

        forward_speed_factor = 1 - np.abs(turn_action) # Couple forward speed with turning
        # Forward movement: overall average of motor neuron potentials, modulated by distance to stimulus
        # This encourages slowing down as the worm approaches the food.
        forward_distance_factor = np.clip(distance_to_stimulus / 20.0, 0.05, 1.0) # Scale factor, min 0.05 to ensure final approach
        move_forward_signal = base_forward_signal * forward_distance_factor # No constant bias
        move_forward_signal = np.clip(move_forward_signal, 0.0, 1) # Ensure it's within valid range

        # Modulate turn_action by motor neuron activity (e.g., overall motor activity could scale turning)
        # This allows the network to learn to activate motor neurons that support the desired turn.
        # For simplicity, let's just use the calculated turn_action for now and let STDP learn the weights.

        # Forward movement: overall average of motor neuron potentials, with a stronger bias
        # This ensures consistent forward movement towards the food.
        move_forward_signal = base_forward_signal + 0.5 # Increased constant bias for forward movement
        move_forward_signal = np.clip(move_forward_signal, 0.1, 1) # Ensure it's within valid range and has a minimum
        
        action = np.array([
            np.clip(turn_action, 0, 1),
            np.clip(-turn_action, 0, 1),
            np.clip(move_forward_signal, 0, 1)
        ], dtype=np.float32)

        return action

    def update_weights(self, reward):
        # Apply more targeted neuromodulated STDP
        # Apply neuromodulated STDP to all eligible synapses
        eligible_synapses = self.eligibility_trace != 0
        if np.any(eligible_synapses):
            if reward > 0:
                delta_w = self.learning_rate * reward * self.eligibility_trace[eligible_synapses]
                self.connectome[eligible_synapses] += delta_w
            elif reward < 0:
                delta_w = self.learning_rate * reward * self.eligibility_trace[eligible_synapses] * 5 # Punish more strongly
                self.connectome[eligible_synapses] += delta_w

        # Decay eligibility trace
        self.eligibility_trace *= 0.95
        
        # Clip weights to prevent runaway values
        self.connectome = np.clip(self.connectome, -1, 1)