import yaml
from stable_baselines3 import PPO
from worm_rl_project.gym_wrapper.envs.worm_env import WormEnv
from worm_rl_project.gym_wrapper.envs.fly_env import FlyEnv

def train():
    with open("worm_rl_project/config/default.yaml", 'r') as stream:
        config = yaml.safe_load(stream)

    # Instantiate the appropriate gym_wrapper environment (worm or fly)
    if config['rl_hyperparameters']['env'] == 'worm':
        env = WormEnv()
    elif config['rl_hyperparameters']['env'] == 'fly':
        env = FlyEnv()
    else:
        raise ValueError("Invalid environment specified in config")

    # Create and train an RL model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        learning_rate=config['rl_hyperparameters']['learning_rate']
    )
    model.learn(total_timesteps=config['neural_sim_settings']['duration'])

    # Save the trained policy
    model.save(f"worm_rl_project/{config['file_paths']['model_checkpoints']}/ppo_{config['rl_hyperparameters']['env']}")

if __name__ == '__main__':
    train()
