import gym
from gym import spaces
import numpy as np
from worm_rl_project.gym_wrapper.utils import NeuralSimulator, translate_cell_voltages

class FlyEnv(gym.Env):
    def __init__(self, nml_file='worm_rl_project/models/fly_full.nml'):
        super(FlyEnv, self).__init__()
        self.simulator = NeuralSimulator(nml_file)
        self.simulator.load_model()

        # Define action and observation spaces
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(10,), dtype=np.float32)

    def step(self, action):
        if action is not None:
            self.simulator.inject_sensory_inputs(action)

        self.simulator.advance_neural_state()
        motor_outputs = self.simulator.read_motor_outputs()
        
        muscle_activations = translate_cell_voltages(motor_outputs)

        observation = motor_outputs
        reward = np.sum(muscle_activations)
        done = False 

        return observation, reward, done, {}

    def reset(self):
        self.simulator.load_model()
        return self.simulator.read_motor_outputs()

    def render(self, mode='human', close=False):
        pass
