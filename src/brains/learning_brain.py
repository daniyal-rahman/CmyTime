import numpy as np
import pandas as pd
from src.brains.base_brain import BaseBrain
from src.models.lif_neuron import LIFNeuron

class LearningBrain(BaseBrain):
    def __init__(self, connectome_path, neuron_types_path, learning_rate=0.01, tau_pre=20, tau_post=20):
        self.connectome_path = connectome_path
        self.connectome, self.neuron_to_idx, self.all_neuron_names = self._load_connectome(connectome_path)
        self.neuron_types = self._categorize_neurons(neuron_types_path)
        
        self.sensory_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['sensory'] if n in self.neuron_to_idx]
        self.dorsal_motor_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['dorsal_motor'] if n in self.neuron_to_idx]
        self.ventral_motor_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['ventral_motor'] if n in self.neuron_to_idx]
        
        # Initialize LIF neurons, each with a spike_output attribute
        self.neurons = [LIFNeuron() for _ in range(len(self.all_neuron_names))]
        for neuron in self.neurons:
            neuron.spike_output = 0.0 # Initialize spike output for the first step

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

        print(f"Initialized LearningBrain with STDP/LIF: {len(self.sensory_neurons_idx)} sensory, {len(self.dorsal_motor_neurons_idx)} dorsal, {len(self.ventral_motor_neurons_idx)} ventral motor neurons")

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
        # Attempt to categorize from the 'NeuronsToMuscle' sheet first
        try:
            xls = pd.ExcelFile(path)
            if 'NeuronsToMuscle' in xls.sheet_names:
                df = pd.read_excel(path, sheet_name='NeuronsToMuscle')
                if 'Function' in df.columns:
                    sensory_neurons = df[df['Function'].str.contains('sensory', case=False, na=False)]['Neuron'].tolist()
                    dorsal_motor_neurons = df[df['Neuron'].str.startswith(('DA', 'DB', 'AS'))]['Neuron'].tolist()
                    ventral_motor_neurons = df[df['Neuron'].str.startswith(('VA', 'VB', 'VD', 'VC'))]['Neuron'].tolist()
                    if sensory_neurons or dorsal_motor_neurons or ventral_motor_neurons:
                        return {
                            'sensory': sensory_neurons,
                            'dorsal_motor': dorsal_motor_neurons,
                            'ventral_motor': ventral_motor_neurons
                        }
        except Exception:
            pass

        # Fallback logic based on neuron names
        df = pd.read_excel(self.connectome_path, sheet_name='Sheet1')
        all_neurons = pd.concat([df['Neuron 1'], df['Neuron 2']]).unique()
        sensory_neurons = [n for n in all_neurons if 'S' in n or n.startswith(('ADE', 'ADF', 'ADL', 'AQR', 'ASE', 'ASG', 'ASI', 'ASJ', 'ASK', 'AWA', 'AWB', 'AWC', 'BAG', 'CEP', 'FLP', 'IL1', 'IL2', 'OLQ', 'PHA', 'PHB', 'PQR', 'URX'))]
        
        dorsal_motor = [
            'DA1', 'DA2', 'DA3', 'DA4', 'DA5', 'DA6', 'DA7', 'DA8', 'DA9',
            'DB1', 'DB2', 'DB3', 'DB4', 'DB5', 'DB6', 'DB7',
            'AS1', 'AS2', 'AS3', 'AS4', 'AS5', 'AS6', 'AS7', 'AS8', 'AS9', 'AS10', 'AS11'
        ]
        ventral_motor = [
            'VA1', 'VA2', 'VA3', 'VA4', 'VA5', 'VA6', 'VA7', 'VA8', 'VA9', 'VA10', 'VA11', 'VA12',
            'VB1', 'VB2', 'VB3', 'VB4', 'VB5', 'VB6', 'VB7', 'VB8', 'VB9', 'VB10', 'VB11',
            'VC1', 'VC2', 'VC3', 'VC4', 'VC5', 'VC6',
            'VD1', 'VD2', 'VD3', 'VD4', 'VD5', 'VD6', 'VD7', 'VD8', 'VD9', 'VD10', 'VD11', 'VD12', 'VD13'
        ]
        
        return {
            'sensory': list(set(sensory_neurons)),
            'dorsal_motor': dorsal_motor,
            'ventral_motor': ventral_motor
        }

    def get_action(self, observation):
        self.current_time += 1.0

        # 1. Calculate External Input Current for Sensory Neurons
        input_current = np.zeros(len(self.all_neuron_names))
        num_sensory = len(self.sensory_neurons_idx)
        
        if num_sensory > 0:
            # Food gradient influences first half of sensory neurons
            food_sensory_input_x = observation['gradient_x'] * 10.0
            food_sensory_input_y = observation['gradient_y'] * 10.0
            for i, idx in enumerate(self.sensory_neurons_idx):
                if i < num_sensory / 2:
                    input_current[idx] += food_sensory_input_x
                else:
                    input_current[idx] += food_sensory_input_y

            # Trap gradient influences a different subset of sensory neurons
            if 'distance_to_nearest_trap' in observation and num_sensory > 2:
                trap_gradient = -1 / (1 + observation['distance_to_nearest_trap'])
                trap_sensory_input_x = trap_gradient * np.cos(observation['relative_trap_angle']) * 10.0
                trap_sensory_input_y = trap_gradient * np.sin(observation['relative_trap_angle']) * 10.0
                # Apply to a different set of sensory neurons (e.g., the last two)
                input_current[self.sensory_neurons_idx[-1]] += trap_sensory_input_x
                input_current[self.sensory_neurons_idx[-2]] += trap_sensory_input_y

        # 2. Add Recurrent Input from Previous Spikes
        previous_spikes = np.array([neuron.spike_output for neuron in self.neurons])
        recurrent_input = self.connectome.T @ previous_spikes
        input_current += recurrent_input

        # 3. Step Each LIF Neuron and Get Current Spikes
        current_spikes = np.zeros(len(self.all_neuron_names))
        for i, neuron in enumerate(self.neurons):
            current_spikes[i] = neuron.step(input_current[i], self.current_time)
            neuron.spike_output = current_spikes[i]

        # 4. Update STDP Traces
        self.pre_synaptic_trace *= np.exp(-1.0 / self.tau_pre)
        self.post_synaptic_trace *= np.exp(-1.0 / self.tau_post)
        self.pre_synaptic_trace[current_spikes == 1.0] = 1.0
        self.post_synaptic_trace[current_spikes == 1.0] = 1.0
        
        self.eligibility_trace += np.outer(self.pre_synaptic_trace, current_spikes)
        self.eligibility_trace -= np.outer(current_spikes, self.post_synaptic_trace)

        # 5. Generate Action from Motor Neuron Membrane Potentials
        dorsal_potentials = np.array([self.neurons[idx].membrane_potential for idx in self.dorsal_motor_neurons_idx])
        ventral_potentials = np.array([self.neurons[idx].membrane_potential for idx in self.ventral_motor_neurons_idx])

        dorsal_activation = np.mean(dorsal_potentials) if len(dorsal_potentials) > 0 else 0.0
        ventral_activation = np.mean(ventral_potentials) if len(ventral_potentials) > 0 else 0.0
        
        # Normalize and clip activations to create the action
        action = np.array([
            np.clip(dorsal_activation, 0, 1),
            np.clip(ventral_activation, 0, 1)
        ], dtype=np.float32)

        return action

    def update_weights(self, reward):
        # Apply neuromodulated STDP
        eligible_synapses = self.eligibility_trace != 0
        if np.any(eligible_synapses):
            # Punish negative rewards more strongly to encourage avoidance
            reward_factor = 5.0 if reward < 0 else 1.0
            delta_w = self.learning_rate * reward * reward_factor * self.eligibility_trace[eligible_synapses]
            self.connectome[eligible_synapses] += delta_w

        # Decay eligibility trace
        self.eligibility_trace *= 0.95
        
        # Clip weights to prevent runaway values
        self.connectome = np.clip(self.connectome, -1, 1)