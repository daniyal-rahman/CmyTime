import unittest
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from worm_rl_project.gym_wrapper.utils import NeuralSimulator
import numpy as np

class TestNeuralSimulator(unittest.TestCase):

    def setUp(self):
        self.lems_file = '/Users/danirahman/Repos/CmyTime/examples/LEMS_c302_A_Muscles.xml'
        self.simulator = NeuralSimulator(self.lems_file)

    def test_advance_and_read(self):
        # Advance the simulation for a few steps
        for _ in range(3):
            self.simulator.advance_neural_state(dt=0.05)

        # Read the motor outputs
        motor_outputs = self.simulator.read_motor_outputs()

        # Assert that the motor outputs are not empty and have the correct shape
        self.assertIsNotNone(motor_outputs)
        self.assertEqual(motor_outputs.shape, (18,))

if __name__ == '__main__':
    unittest.main()
