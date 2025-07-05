
from .engine import Engine

class Runner:
    def __init__(self, engine, num_steps, environment=None):
        self.engine = engine
        self.num_steps = num_steps
        self.environment = environment
        self.rewards = []
        self.positions = []
        self.weight_history = [] # New: to store sum of weights at each step

    def run(self):
        if self.environment:
            self.environment.reset()
            self.positions.append(self.environment.current_position.copy())

        for _ in range(self.num_steps):
            spikes, reward, done = self.engine.step()
            self.rewards.append(reward)
            self.weight_history.append(self.engine.connectome.sum()) # Record weight sum
            if self.environment:
                self.positions.append(self.environment.current_position.copy())
            if done:
                break
