import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.core.base_env import BaseEnv


class ChemoattractionEnv(BaseEnv):
    def __init__(self, stimulus_location=(100, 100), initial_position=(0, 0), initial_angle=0.0, movement_speed=10.0, rotation_scale=1.0):
        super(ChemoattractionEnv, self).__init__()
        self.stimulus_location = np.array(stimulus_location, dtype=np.float32)
        self.initial_position = np.array(initial_position, dtype=np.float32)
        self.initial_angle = initial_angle
        self.movement_speed = movement_speed
        self.rotation_scale = rotation_scale

        self.action_space = spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'current_position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'stimulus_location': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'distance_to_stimulus': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'relative_stimulus_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            'gradient_x': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'gradient_y': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_position = self.initial_position.copy()
        self.current_angle = self.initial_angle
        self.positions_history = [self.current_position.copy()]
        self.prev_distance_to_stimulus = np.linalg.norm(self.current_position - self.stimulus_location)
        return self._get_obs(), self._get_info()

    def step(self, action):
        turn_left, turn_right, move_forward = action

        self.current_angle += (turn_left - turn_right) * self.rotation_scale
        delta_x = move_forward * self.movement_speed * np.cos(self.current_angle)
        delta_y = move_forward * self.movement_speed * np.sin(self.current_angle)
        self.current_position += np.array([delta_x, delta_y])

        self.positions_history.append(self.current_position.copy())

        distance_to_stimulus = np.linalg.norm(self.current_position - self.stimulus_location)
        reward = self.prev_distance_to_stimulus - distance_to_stimulus
        self.prev_distance_to_stimulus = distance_to_stimulus

        terminated = False 
        truncated = False
        
        return self._get_obs(), reward, terminated, truncated, self._get_info()

    def _get_obs(self):
        vec_to_stimulus = self.stimulus_location - self.current_position
        dist_to_stimulus = np.linalg.norm(vec_to_stimulus)

        stimulus_angle = np.arctan2(vec_to_stimulus[1], vec_to_stimulus[0])
        relative_stimulus_angle = stimulus_angle - self.current_angle
        relative_stimulus_angle = np.arctan2(np.sin(relative_stimulus_angle), np.cos(relative_stimulus_angle))

        gradient_x = np.cos(relative_stimulus_angle)
        gradient_y = np.sin(relative_stimulus_angle)

        return {
            'current_position': self.current_position.copy(),
            'stimulus_location': self.stimulus_location.copy(),
            'distance_to_stimulus': np.array([dist_to_stimulus], dtype=np.float32),
            'relative_stimulus_angle': np.array([relative_stimulus_angle], dtype=np.float32),
            'gradient_x': np.array([gradient_x], dtype=np.float32),
            'gradient_y': np.array([gradient_y], dtype=np.float32),
        }

    def _get_info(self):
        return {'positions_history': self.positions_history}