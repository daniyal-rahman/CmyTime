from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
from .base_env import BaseEnv


actions = [0, 1]


@dataclass
class CandyButtonEnv(BaseEnv):
    reward_delay: int = 3
    correct_action: int = 0
    queue: list = field(default_factory=list)
    last_obs: np.ndarray = field(default_factory=lambda: np.zeros(2))

    def reset(self):
        self.queue.clear()
        self.correct_action = np.random.choice(actions)
        self.last_obs = np.zeros(2)
        return self.observation()

    def step(self, action: int):
        self.queue.append((self.reward_delay, action))
        self.last_obs = np.eye(2)[action]
        reward = self._update_queue()
        return self.observation(), reward

    def _update_queue(self) -> float:
        reward = 0.0
        new_queue = []
        for delay, act in self.queue:
            delay -= 1
            if delay <= 0:
                reward += 1.0 if act == self.correct_action else 0.0
            else:
                new_queue.append((delay, act))
        self.queue = new_queue
        return reward

    def observation(self):
        return self.last_obs

    def reward(self) -> float:
        # no additional reward outside of step
        return 0.0
