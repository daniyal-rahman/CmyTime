import numpy as np
import gymnasium as gym
from gymnasium import spaces

from src.core.base_env import BaseEnv

class TrappedChemoattractionEnv(BaseEnv):
    def __init__(self, stimulus_location=(100, 100), initial_position=(0, 0), initial_angle=0.0, movement_speed=10.0, rotation_scale=1.0):
        super(TrappedChemoattractionEnv, self).__init__()
        self.stimulus_location = np.array(stimulus_location, dtype=np.float32)
        self.initial_position = np.array(initial_position, dtype=np.float32)
        self.initial_angle = initial_angle
        self.movement_speed = movement_speed
        self.rotation_scale = rotation_scale

        self.traps = [
            {'location': np.array([50, 50]), 'radius': 10, 'penalty': -50.0},
            {'location': np.array([25, 75]), 'radius': 10, 'penalty': -50.0},
            {'location': np.array([75, 25]), 'radius': 10, 'penalty': -50.0},
        ]

        self.action_space = spaces.Box(low=np.array([-1, -1, 0]), high=np.array([1, 1, 1]), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'current_position': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'stimulus_location': spaces.Box(low=-np.inf, high=np.inf, shape=(2,), dtype=np.float32),
            'distance_to_stimulus': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'relative_stimulus_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
            'gradient_x': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'gradient_y': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32),
            'distance_to_nearest_trap': spaces.Box(low=0, high=np.inf, shape=(1,), dtype=np.float32),
            'relative_trap_angle': spaces.Box(low=-np.pi, high=np.pi, shape=(1,), dtype=np.float32),
        })

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_position = self.initial_position.copy()
        self.current_angle = self.initial_angle
        self.positions_history = [self.current_position.copy()]
        self.prev_distance_to_stimulus = np.linalg.norm(self.current_position - self.stimulus_location)
        self.time_in_trap = 0
        self.total_trap_penalty = 0.0
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

        # Add penalty for being in a trap
        in_trap = False
        for trap in self.traps:
            if np.linalg.norm(self.current_position - trap['location']) < trap['radius']:
                reward += trap['penalty']
                self.total_trap_penalty += trap['penalty']
                in_trap = True
        
        if in_trap:
            self.time_in_trap += 1

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

        # Get info about the nearest trap
        distances_to_traps = [np.linalg.norm(self.current_position - trap['location']) for trap in self.traps]
        nearest_trap_idx = np.argmin(distances_to_traps)
        nearest_trap = self.traps[nearest_trap_idx]
        
        vec_to_nearest_trap = nearest_trap['location'] - self.current_position
        dist_to_nearest_trap = np.linalg.norm(vec_to_nearest_trap)
        
        trap_angle = np.arctan2(vec_to_nearest_trap[1], vec_to_nearest_trap[0])
        relative_trap_angle = trap_angle - self.current_angle
        relative_trap_angle = np.arctan2(np.sin(relative_trap_angle), np.cos(relative_trap_angle))

        return {
            'current_position': self.current_position.copy(),
            'stimulus_location': self.stimulus_location.copy(),
            'distance_to_stimulus': np.array([dist_to_stimulus], dtype=np.float32),
            'relative_stimulus_angle': np.array([relative_stimulus_angle], dtype=np.float32),
            'gradient_x': np.array([gradient_x], dtype=np.float32),
            'gradient_y': np.array([gradient_y], dtype=np.float32),
            'distance_to_nearest_trap': np.array([dist_to_nearest_trap], dtype=np.float32),
            'relative_trap_angle': np.array([relative_trap_angle], dtype=np.float32),
        }

    def _get_info(self):
        return {
            'positions_history': self.positions_history,
            'time_in_trap': self.time_in_trap,
            'total_trap_penalty': self.total_trap_penalty
        }

    def render(self, mode='human'):
        pass

    def close(self):
        pass
