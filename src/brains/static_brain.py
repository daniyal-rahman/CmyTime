import numpy as np
import pandas as pd
from src.brains.base_brain import BaseBrain

class StaticBrain(BaseBrain):
    def __init__(self, connectome_path, neuron_types_path):
        self.connectome_path = connectome_path
        self.connectome, self.neuron_to_idx, self.all_neuron_names = self._load_connectome(connectome_path)
        self.neuron_types = self._categorize_neurons(neuron_types_path)
        
        self.sensory_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['sensory'] if n in self.neuron_to_idx]
        self.dorsal_motor_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['dorsal_motor'] if n in self.neuron_to_idx]
        self.ventral_motor_neurons_idx = [self.neuron_to_idx[n] for n in self.neuron_types['ventral_motor'] if n in self.neuron_to_idx]
        
        print(f"Initialized StaticBrain: {len(self.sensory_neurons_idx)} sensory, {len(self.dorsal_motor_neurons_idx)} dorsal motor, {len(self.ventral_motor_neurons_idx)} ventral motor neurons")

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
                    # This is a simplification; real mapping is more complex.
                    dorsal_motor_neurons = df[df['Neuron'].str.startswith(('DA', 'DB', 'AS'))]['Neuron'].tolist()
                    ventral_motor_neurons = df[df['Neuron'].str.startswith(('VA', 'VB', 'VD', 'VC'))]['Neuron'].tolist()
                    if sensory_neurons or dorsal_motor_neurons or ventral_motor_neurons:
                        return {
                            'sensory': sensory_neurons,
                            'dorsal_motor': dorsal_motor_neurons,
                            'ventral_motor': ventral_motor_neurons
                        }
        except Exception:
            pass # Fallback to heuristic categorization if file reading fails

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
        """
        Generates an action based on sensory input driving motor neuron activation.
        The action is defined by the differential activation of dorsal and ventral motor neurons.
        """
        # 1. Calculate Sensory Input
        sensory_input = np.zeros(len(self.all_neuron_names))
        
        # Simplified sensory mapping: gradient influences a subset of sensory neurons
        # A more biologically accurate model would map specific sensory neurons to specific stimuli.
        # For now, we use the gradient to simulate a general chemo-attractive signal.
        num_sensory = len(self.sensory_neurons_idx)
        if num_sensory > 0:
            # Distribute gradient info across sensory neurons
            # This is a heuristic to ensure varied input
            for i, idx in enumerate(self.sensory_neurons_idx):
                if i < num_sensory / 2:
                    sensory_input[idx] = observation['gradient_x']
                else:
                    sensory_input[idx] = observation['gradient_y']

        # 2. Propagate Sensory Input through the Connectome
        # This simulates the neural activity propagating from sensory to motor neurons.
        # Note: This is a single-step propagation (rate-based model), not a dynamic simulation.
        all_neuron_activations = self.connectome.T @ sensory_input

        # 3. Determine Muscle Activations from Motor Neurons
        # The activation of dorsal and ventral muscles is the average activation of their respective motor neurons.
        dorsal_activation = 0.0
        if self.dorsal_motor_neurons_idx:
            dorsal_activation = np.mean(all_neuron_activations[self.dorsal_motor_neurons_idx])

        ventral_activation = 0.0
        if self.ventral_motor_neurons_idx:
            ventral_activation = np.mean(all_neuron_activations[self.ventral_motor_neurons_idx])
            
        # 4. Normalize and Create Action
        # The action is a 2-element array: [dorsal_activation, ventral_activation]
        # We clip the values to be within a reasonable range, e.g., [0, 1]
        action = np.array([
            np.clip(dorsal_activation, 0, 1),
            np.clip(ventral_activation, 0, 1)
        ], dtype=np.float32)

        return action