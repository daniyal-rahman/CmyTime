import gymnasium as gym
from gymnasium import spaces
import numpy as np
from worm_rl_project.gym_wrapper.utils import NeuralSimulator, translate_cell_voltages
from worm_rl_project.gym_wrapper.sibernetic_c302 import run_sibernetic
import os

class WormEnv(gym.Env):
    """
    Custom Gym environment for the C. elegans locomotion task.
    """
    metadata = {'render_modes': ['human']}

    def __init__(self, nml_file, env_dt=0.01, neural_dt=0.0001):
        super(WormEnv, self).__init__()
        
        # Simulation parameters
        self.nml_file = nml_file
        self.env_dt = env_dt
        self.neural_dt = neural_dt
        self.neural_steps_per_env_step = int(self.env_dt / self.neural_dt)

        # Initialize simulators
        self.simulator = NeuralSimulator(self.nml_file)
        
        # Get the number of motor neurons from the simulator
        num_motor_neurons = len(self.simulator._get_motor_neuron_names())

        # Define action and observation spaces
        # Action space: direct override of muscle activations
        self.action_space = spaces.Box(low=0, high=1, shape=(num_motor_neurons,), dtype=np.float32)
        
        # Observation space: current muscle activations
        self.observation_space = spaces.Box(low=0, high=1, shape=(num_motor_neurons,), dtype=np.float32)
        
        self.reward_range = (-np.inf, np.inf)
        
        # State variables
        self.current_position = np.zeros(3) # x, y, z from Sibernetic

    def step(self, action):
        # If action is provided, it overrides the CPG output
        # For now, we let the CPG drive the worm and ignore the 'action' parameter.
        
        # Advance the neural simulation with fine-grained steps
        for _ in range(self.neural_steps_per_env_step):
            self.simulator.advance_neural_state(dt_s=self.neural_dt)
        
        motor_outputs = self.simulator.read_motor_outputs()
        muscle_activations = translate_cell_voltages(motor_outputs)

        # Run the physics simulation for one environment timestep
        new_position = run_sibernetic(
            muscle_activations, 
            dt_s=self.env_dt, 
            sim_duration_s=self.env_dt
        )

        observation = muscle_activations
        
        # Reward is the forward movement (change in x-position)
        reward = float(new_position[0] - self.current_position[0])
        self.current_position = new_position
        
        # For now, the episode doesn't terminate on its own
        terminated = False 
        truncated = False

        return observation, reward, terminated, truncated, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        
        # Reset the neural simulator to its initial state
        self.simulator = NeuralSimulator(self.nml_file)
        
        # Reset the physics state
        # NOTE: The current `run_sibernetic` implementation starts a fresh
        # simulation at every step, so it's implicitly reset. If this changes
        # to a persistent process, an explicit reset call will be needed here.
        self.current_position = np.zeros(3)
        
        # Return the initial observation
        motor_outputs = self.simulator.read_motor_outputs()
        initial_observation = translate_cell_voltages(motor_outputs)
        
        return initial_observation, {}

    def render(self, mode='human'):
        # Placeholder for rendering
        pass

    def close(self):
        # Clean up any resources if needed
        pass
