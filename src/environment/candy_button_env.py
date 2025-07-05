
from .base_env import BaseEnv

class CandyButtonEnv(BaseEnv):
    def __init__(self, delay):
        self.delay = delay
        self.action_queue = []

    def reset(self):
        self.action_queue = []

    def step(self, action):
        self.action_queue.append(action)

    def get_observation(self):
        return None

    def get_reward(self):
        if len(self.action_queue) >= self.delay:
            return 1 if self.action_queue.pop(0) == 'A' else 0
        return 0
