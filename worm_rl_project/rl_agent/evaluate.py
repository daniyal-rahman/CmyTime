import yaml
from stable_baselines3 import PPO
from worm_rl_project.gym_wrapper.envs.worm_env import WormEnv
from worm_rl_project.gym_wrapper.envs.fly_env import FlyEnv

def evaluate():
    with open("worm_rl_project/config/default.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Instantiate the appropriate gym_wrapper environment (worm or fly)
    if config['rl_hyperparameters']['env'] == 'worm':
        env = WormEnv()
    elif config['rl_hyperparameters']['env'] == 'fly':
        env = FlyEnv()
    else:
        raise ValueError("Invalid environment specified in config")

    # Load a saved policy
    model = PPO.load(f"worm_rl_project/{config['file_paths']['model_checkpoints']}/ppo_{config['rl_hyperparameters']['env']}")

    # Run a fixed number of evaluation episodes
    obs = env.reset()
    for _ in range(config['neural_sim_settings']['duration']):
        action, _states = model.predict(obs)
        obs, rewards, dones, info = env.step(action)
        env.render()

if __name__ == '__main__':
    evaluate()
