
import gym
import pytest
from worm_rl_project.gym_wrapper.envs.worm_env import WormEnv
from worm_rl_project.gym_wrapper.envs.fly_env import FlyEnv
import numpy as np

def test_worm_env_init():
    env = WormEnv()
    assert isinstance(env, gym.Env)

def test_worm_env_step():
    env = WormEnv()
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, object)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

def test_fly_env_init():
    env = FlyEnv()
    assert isinstance(env, gym.Env)

def test_fly_env_step():
    env = FlyEnv()
    obs = env.reset()
    action = env.action_space.sample()
    obs, reward, done, info = env.step(action)
    assert isinstance(obs, object)
    assert isinstance(reward, float)
    assert isinstance(done, bool)

def test_worm_forward_movement():
    env = WormEnv()
    env.reset()
    initial_position = env.x_position
    action = np.array([1.0]) # Apply a constant forward force
    obs, reward, done, info = env.step(action)
    assert env.x_position > initial_position
