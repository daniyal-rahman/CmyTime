import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from worm_rl_project.gym_wrapper.envs.worm_env import WormEnv
import numpy as np

class TestWormEnv(unittest.TestCase):

    def setUp(self):
        self.lems_file = '/Users/danirahman/Repos/CmyTime/examples/LEMS_c302_A_Muscles.xml'
        self.env = WormEnv(nml_file=self.lems_file)

    def test_step(self):
        observation = self.env.reset()
        self.assertIsInstance(observation, np.ndarray)
        self.assertEqual(observation.shape, (18,))

        for _ in range(3):
            action = self.env.action_space.sample()
            observation, reward, done, info = self.env.step(action)

            self.assertIsInstance(observation, np.ndarray)
            self.assertEqual(observation.shape, (18,))
            self.assertIsInstance(reward, float)
            self.assertIsInstance(done, bool)

if __name__ == '__main__':
    unittest.main()
