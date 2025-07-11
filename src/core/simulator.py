class Simulator:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.positions = []

    def run(self, num_steps):
        obs, info = self.env.reset()
        self.positions.append(info['positions_history'][-1])
        for _ in range(num_steps):
            action = self.agent.act(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            self.positions.append(info['positions_history'][-1])
            if terminated or truncated:
                obs, info = self.env.reset()