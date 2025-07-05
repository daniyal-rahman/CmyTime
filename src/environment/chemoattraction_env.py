
import numpy as np
from .base_env import BaseEnv

class ChemoattractionEnv(BaseEnv):
    def __init__(self, stimulus_location=(10, 10), initial_position=(0, 0), initial_angle=0.0, movement_speed=0.1, rotation_scale=0.1):
        self.stimulus_location = np.array(stimulus_location, dtype=float)
        self.initial_position = np.array(initial_position, dtype=float) # Corrected: added initial_position
        self.current_position = np.copy(self.initial_position)
        self.initial_angle = initial_angle # Corrected: added initial_angle
        self.current_angle = initial_angle # Radians
        self.movement_speed = movement_speed
        self.rotation_scale = rotation_scale
        self.positions_history = []

    def reset(self):
        self.current_position = np.copy(self.initial_position)
        self.current_angle = self.initial_angle # Reset angle as well
        self.positions_history = [self.current_position.copy()]
        return self.get_observation()

    def step(self, motor_actions): # motor_actions: [turn_left_signal, turn_right_signal, move_forward_signal]
        # Interpret motor actions
        turn_left = motor_actions[0]
        turn_right = motor_actions[1]
        move_forward = motor_actions[2]

        # Update angle based on turn signals
        self.current_angle += (turn_left - turn_right) * self.rotation_scale

        # Update position based on current angle and forward movement
        delta_x = move_forward * self.movement_speed * np.cos(self.current_angle)
        delta_y = move_forward * self.movement_speed * np.sin(self.current_angle)
        self.current_position += np.array([delta_x, delta_y])

        self.positions_history.append(self.current_position.copy())

        distance_to_stimulus = np.linalg.norm(self.current_position - self.stimulus_location)
        reward = -distance_to_stimulus # Negative reward for distance

        done = False
        info = {}

        return self.get_observation(), reward, done, info

    def get_observation(self):
        # Calculate vector from worm to stimulus
        vec_to_stimulus = self.stimulus_location - self.current_position
        dist_to_stimulus = np.linalg.norm(vec_to_stimulus)

        # Calculate angle to stimulus relative to worm's current heading
        stimulus_angle = np.arctan2(vec_to_stimulus[1], vec_to_stimulus[0])
        relative_stimulus_angle = stimulus_angle - self.current_angle

        # Normalize angle to [-pi, pi]
        relative_stimulus_angle = np.arctan2(np.sin(relative_stimulus_angle), np.cos(relative_stimulus_angle))

        # Provide directional gradient as observation (e.g., sine and cosine of relative angle)
        # This allows the network to infer direction without explicit angle processing
        gradient_x = np.cos(relative_stimulus_angle)
        gradient_y = np.sin(relative_stimulus_angle)

        return {
            'current_position': self.current_position.copy(),
            'stimulus_location': self.stimulus_location.copy(),
            'distance_to_stimulus': dist_to_stimulus,
            'relative_stimulus_angle': relative_stimulus_angle,
            'gradient_x': gradient_x,
            'gradient_y': gradient_y
        }
