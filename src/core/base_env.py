import gymnasium as gym

class BaseEnv(gym.Env):
    def reset(self, seed=None, options=None):
        pass

    def step(self, action):
        raise NotImplementedError

    def get_observation(self):
        raise NotImplementedError

    def get_reward(self):
        raise NotImplementedError
