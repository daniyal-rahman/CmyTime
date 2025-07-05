
import numpy as np

class Engine:
    def __init__(self, connectome, neurons, plasticity_rule, environment=None):
        self.connectome = connectome
        self.neurons = neurons
        self.plasticity_rule = plasticity_rule
        self.environment = environment
        self.current_time = 0

    def step(self):
        input_current = np.zeros(len(self.neurons))

        if self.environment:
            observation = self.environment.get_observation()
            # Map gradient observations to sensory neurons (e.g., first 5 neurons)
            # Distribute the gradient signal across multiple neurons
            if 'gradient_x' in observation and 'gradient_y' in observation:
                for i in range(5):
                    input_current[i] = observation['gradient_x'] * (50.0 + i*5.0) # Increased varying scale
                    input_current[i+5] = observation['gradient_y'] * (50.0 + i*5.0) # Increased varying scale

        spikes = np.array([neuron.step(current, self.current_time) for neuron, current in zip(self.neurons, input_current)])

        # Motor output: assuming last few neurons control turning and forward movement
        # Use sum of spikes from a group of neurons for each motor action
        motor_actions = np.array([0.0, 0.0, 0.0]) # [turn_left, turn_right, move_forward]
        if self.environment and len(self.neurons) >= 15: # Ensure enough neurons for mapping
            # Sum spikes from a range of neurons for each action
            turn_left_spikes = np.sum(spikes[len(self.neurons)-15:len(self.neurons)-10])
            turn_right_spikes = np.sum(spikes[len(self.neurons)-10:len(self.neurons)-5])
            move_forward_spikes = np.sum(spikes[len(self.neurons)-5:])

            motor_actions[0] = turn_left_spikes
            motor_actions[1] = turn_right_spikes
            motor_actions[2] = move_forward_spikes

            _, reward, done, _ = self.environment.step(motor_actions)
        else:
            reward = 0
            done = False

        # Collect spike times for plasticity rule
        pre_spike_times = np.array([n.last_spike_time for n in self.neurons])
        post_spike_times = np.array([n.last_spike_time for n in self.neurons])

        # Pass reward to plasticity rule for modulation
        self.connectome = self.plasticity_rule.update(self.connectome, pre_spike_times, post_spike_times, self.current_time, reward)
        self.current_time += 1
        return spikes, reward, done
