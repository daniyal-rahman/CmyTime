from __future__ import annotations
from abc import ABC, abstractmethod


class BaseEnv(ABC):
    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step(self, action):
        pass

    @abstractmethod
    def observation(self):
        pass

    @abstractmethod
    def reward(self) -> float:
        pass
