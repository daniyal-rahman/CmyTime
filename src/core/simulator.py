class Simulator:
    def __init__(self, agent, env):
        self.agent = agent
        self.env = env
        self.positions = []

    def run(self, num_steps):
        self.positions = [] # Reset trajectory for each run
        self.time_in_trap_history = []
        self.total_trap_penalty_history = []
        obs, info = self.env.reset()
        self.positions.append(info['positions_history'][-1])
        for _ in range(num_steps):
            action = self.agent.act(obs)
            obs, reward, terminated, truncated, info = self.env.step(action)
            
            if hasattr(self.agent.brain, 'update_weights'):
                self.agent.brain.update_weights(reward)
                
            self.positions.append(info['positions_history'][-1])
            self.time_in_trap_history.append(info['time_in_trap'])
            self.total_trap_penalty_history.append(info['total_trap_penalty'])
            if terminated or truncated:
                obs, info = self.env.reset()