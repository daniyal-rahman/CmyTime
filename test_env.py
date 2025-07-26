import gym
import sys
sys.path.append('/Users/danirahman/Repos/CmyTime/worm_rl_project')
from gym_wrapper.envs.worm_env import WormEnv

def run_test():
    lems_file = '/Users/danirahman/Repos/CmyTime/examples/LEMS_c302_A_Muscles.xml'
    env = WormEnv(nml_file=lems_file)
    
    observation = env.reset()
    
    for i in range(10):
        print(f"Step {i+1}")
        action = env.action_space.sample() # Dummy action
        observation, reward, done, info = env.step(action)
        
        print(f"  Observation: {observation}")
        print(f"  Reward: {reward}")
        print(f"  Done: {done}")

if __name__ == '__main__':
    run_test()
