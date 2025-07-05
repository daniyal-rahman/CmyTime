import numpy as np
from src.environment.candy_button_env import CandyButtonEnv


def test_candy_button_reward_delay():
    env = CandyButtonEnv(reward_delay=2)
    env.reset()
    env.correct_action = 1
    _, r1 = env.step(1)
    assert r1 == 0
    _, r2 = env.step(0)
    assert r2 == 1.0
