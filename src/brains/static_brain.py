import numpy as np
import pandas as pd
from src.brains.base_brain import BaseBrain

class StaticBrain(BaseBrain):
    def __init__(self, connectome_path, neuron_types_path):
        self.connectome_path = connectome_path
        self.connectome, self.neuron_to_idx, self.all_neuron_names = self._load_connectome(connectome_path)
        self.neuron_types = self._categorize_neurons(neuron_types_path)
        
        self.sensory_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['sensory'] if n in self.neuron_to_idx]
        self.motor_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['motor'] if n in self.neuron_to_idx]
        print(f"Initialized StaticBrain: {len(self.sensory_neurons_idx)} sensory neurons, {len(self.motor_neurons_idx)} motor neurons")

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

        # Fallback logic
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
        """
        Generates an action based on sensory input.
        """
        relative_angle = observation['relative_stimulus_angle'][0]

        # Proportional control for turning
        # The farther the angle, the stronger the turn
        turn_action = relative_angle / np.pi  # Normalize to [-1, 1]

        # Modulate forward speed based on alignment
        # Move faster when aligned with the stimulus
        forward_speed_factor = 1 - np.abs(turn_action)
        
        # Use a base forward signal from motor neurons, but modulate it
        sensory_input = np.zeros(len(self.all_neuron_names))
        for i, idx in enumerate(self.sensory_neurons_idx):
            if i % 2 == 0:
                sensory_input[idx] = observation['gradient_x']
            else:
                sensory_input[idx] = observation['gradient_y']
        
        motor_activation = self.connectome.T @ sensory_input
        motor_vals = motor_activation[self.motor_neurons_idx]
        
        base_forward_signal = np.mean(motor_vals) if len(motor_vals) > 0 else 0
        
        move_forward_signal = base_forward_signal * forward_speed_factor

        turn_left = np.clip(turn_action, 0, 1)
        turn_right = np.clip(-turn_action, 0, 1)
        
        action = np.array([
            turn_left,
            turn_right,
            np.clip(move_forward_signal, 0, 1)
        ], dtype=np.float32)

        return action