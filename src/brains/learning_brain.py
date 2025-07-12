import numpy as np
import pandas as pd
from src.brains.base_brain import BaseBrain

class LearningBrain(BaseBrain):
    def __init__(self, connectome_path, neuron_types_path, learning_rate=0.01, tau_pre=20, tau_post=20, spike_threshold=0.9):
        self.connectome_path = connectome_path
        self.connectome, self.neuron_to_idx, self.all_neuron_names = self._load_connectome(connectome_path)
        self.neuron_types = self._categorize_neurons(neuron_types_path)
        
        self.sensory_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['sensory'] if n in self.neuron_to_idx]
        self.motor_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['motor'] if n in self.neuron_to_idx]
        
        # Learning parameters
        self.learning_rate = learning_rate
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.spike_threshold = 0.5 # Decreased spike threshold for easier firing

        # Neuron state variables
        num_neurons = len(self.all_neuron_names)
        self.pre_synaptic_trace = np.zeros(num_neurons)
        self.post_synaptic_trace = np.zeros(num_neurons)
        self.eligibility_trace = np.zeros_like(self.connectome)

        print(f"Initialized LearningBrain with STDP: {len(self.sensory_neurons_idx)} sensory, {len(self.motor_neurons_idx)} motor neurons")

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
        # --- Separate Sensory Processing for Food and Traps ---
        food_sensory_input = np.zeros(len(self.all_neuron_names))
        # Increased scaling for food sensory input
        food_sensory_input[self.sensory_neurons_idx[0]] = observation['gradient_x'] * 5.0
        food_sensory_input[self.sensory_neurons_idx[1]] = observation['gradient_y'] * 5.0

        trap_sensory_input = np.zeros(len(self.all_neuron_names))
        if 'distance_to_nearest_trap' in observation:
            trap_gradient = -1 / (1 + observation['distance_to_nearest_trap'])
            # Increased scaling for trap sensory input
            trap_sensory_input[self.sensory_neurons_idx[2]] = trap_gradient * np.cos(observation['relative_trap_angle']) * 5.0
            trap_sensory_input[self.sensory_neurons_idx[3]] = trap_gradient * np.sin(observation['relative_trap_angle']) * 5.0

        # --- Neuron Activation and Spiking ---
        # Combine activations from food and trap pathways
        food_activation = self.connectome.T @ food_sensory_input
        trap_activation = self.connectome.T @ trap_sensory_input
        
        activation = food_activation + trap_activation
        spikes = activation > self.spike_threshold

        # --- STDP Trace Updates ---
        self.pre_synaptic_trace *= np.exp(-1.0 / self.tau_pre)
        self.post_synaptic_trace *= np.exp(-1.0 / self.tau_post)
        self.pre_synaptic_trace[spikes] = 1.0
        self.post_synaptic_trace[spikes] = 1.0
        self.eligibility_trace += np.outer(self.pre_synaptic_trace, spikes)
        self.eligibility_trace -= np.outer(spikes, self.post_synaptic_trace)

        # --- Action Generation ---
        motor_activation = activation[self.motor_neurons_idx]
        
        # Food-seeking behavior
        relative_angle = observation['relative_stimulus_angle'][0]
        turn_action = relative_angle / np.pi
        
        # Trap-avoidance behavior
        if 'distance_to_nearest_trap' in observation and observation['distance_to_nearest_trap'] < 20:
             # If close to a trap, turn away from it (increased strength)
            turn_action += -observation['relative_trap_angle'][0] / np.pi * 2.0 # Increased turning strength

        forward_speed_factor = 1 - np.abs(turn_action)
        base_forward_signal = np.mean(motor_activation) if len(motor_activation) > 0 else 0
        move_forward_signal = base_forward_signal * forward_speed_factor
        
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