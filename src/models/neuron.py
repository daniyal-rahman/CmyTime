from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np


def clamp(x: np.ndarray, min_value: float, max_value: float) -> np.ndarray:
    return np.minimum(np.maximum(x, min_value), max_value)


@dataclass
class Neuron:
    """Simple neuron with membrane potential."""

    v: float = 0.0
    threshold: float = 1.0
    reset: float = 0.0

    def step(self, input_current: float) -> int:
        """Advance neuron state by one step.

        Args:
            input_current: Input current for this step.

        Returns:
            Spike (1 if threshold crossed else 0).
        """
        self.v += input_current
        if self.v >= self.threshold:
            self.v = self.reset
            return 1
        return 0

    def reset_state(self) -> None:
        self.v = self.reset
