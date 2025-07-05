from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .neuron import Neuron


@dataclass
class LIFNeuron(Neuron):
    tau: float = 20.0  # membrane time constant
    dt: float = 1.0

    def step(self, input_current: float) -> int:
        dv = (-self.v / self.tau + input_current) * self.dt
        self.v += dv
        if self.v >= self.threshold:
            self.v = self.reset
            return 1
        return 0
