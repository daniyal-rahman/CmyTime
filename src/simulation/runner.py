
from .engine import Engine

class Runner:
    def __init__(self, engine, num_steps):
        self.engine = engine
        self.num_steps = num_steps

    def run(self):
        for _ in range(self.num_steps):
            self.engine.step()
