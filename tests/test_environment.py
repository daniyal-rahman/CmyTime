
import numpy as np
from src.environment.chemoattraction_env import ChemoattractionEnv

def test_chemoattraction_env_reset():
    env = ChemoattractionEnv(stimulus_location=(10, 10), initial_position=(1, 1))
    obs = env.reset()
    assert np.array_equal(obs['current_position'], np.array([1, 1]))
    assert np.array_equal(obs['stimulus_location'], np.array([10, 10]))
    assert np.array_equal(obs['relative_position'], np.array([9, 9]))
    assert len(env.positions_history) == 1

def test_chemoattraction_env_step():
    env = ChemoattractionEnv(stimulus_location=(10, 10), initial_position=(0, 0), movement_scale=1.0)
    env.reset()

    # Move towards stimulus
    obs, reward, done, info = env.step(np.array([1, 1]))
    assert np.array_equal(obs['current_position'], np.array([1, 1]))
    assert reward < 0  # Reward should be negative (distance)
    assert not done
    assert len(env.positions_history) == 2

    # Move further towards stimulus
    obs, reward2, done, info = env.step(np.array([1, 1]))
    assert np.array_equal(obs['current_position'], np.array([2, 2]))
    assert reward2 > reward  # Reward should increase (less negative) as distance decreases

def test_chemoattraction_env_reward_calculation():
    env = ChemoattractionEnv(stimulus_location=(0, 0), initial_position=(10, 0), movement_scale=1.0)
    env.reset()

    # Initial distance is 10, reward should be -10
    obs, reward, done, info = env.step(np.array([0, 0])) # No movement
    assert reward == -10.0

    # Move to (5,0), distance 5, reward -5
    obs, reward, done, info = env.step(np.array([-5, 0]))
    assert reward == -5.0

    # Move to (0,0), distance 0, reward 0
    obs, reward, done, info = env.step(np.array([-5, 0]))
    assert reward == 0.0
