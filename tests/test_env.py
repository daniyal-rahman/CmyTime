
from src.environment.candy_button_env import CandyButtonEnv

def test_candy_button_env():
    env = CandyButtonEnv(delay=2)
    env.step('A')
    env.step('B')
    assert env.get_reward() == 1
    assert env.get_reward() == 0
