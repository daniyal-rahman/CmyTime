import pytest
import yaml
from worm_rl_project.rl_agent import train, evaluate
import os

def test_train_and_evaluate():
    # To speed up the test, I'll create a temporary config file with a short duration
    test_config = {
        'neural_sim_settings': {
            'dt': 0.01,
            'duration': 10,
            'parameter_set': 'A'
        },
        'physics_sim_settings': {
            'timestep': 0.01,
            'viscosity': 1.0
        },
        'rl_hyperparameters': {
            'env': 'worm',
            'algorithm': 'PPO',
            'network_size': [8, 8],
            'learning_rate': 0.0003
        },
        'file_paths': {
            'neuroml': 'models/',
            'model_checkpoints': 'models/'
        }
    }
    with open("tests/rl_agent/test_config.yaml", 'w') as f:
        yaml.dump(test_config, f)

    # I'll also need to temporarily replace the original config file path
    # in the train and evaluate scripts.
    with open("worm_rl_project/rl_agent/train.py", "r") as f:
        train_script = f.read()
    
    with open("worm_rl_project/rl_agent/evaluate.py", "r") as f:
        evaluate_script = f.read()

    # Replace the config file path
    train_script = train_script.replace(
        'worm_rl_project/config/default.yaml',
        'tests/rl_agent/test_config.yaml'
    )
    evaluate_script = evaluate_script.replace(
        'worm_rl_project/config/default.yaml',
        'tests/rl_agent/test_config.yaml'
    )

    # Write the modified scripts to temporary files
    with open("tests/rl_agent/temp_train.py", "w") as f:
        f.write(train_script)
    
    with open("tests/rl_agent/temp_evaluate.py", "w") as f:
        f.write(evaluate_script)

    # Run the training and evaluation
    os.system("python tests/rl_agent/temp_train.py")
    os.system("python tests/rl_agent/temp_evaluate.py")

    # Clean up the temporary files
    os.remove("tests/rl_agent/test_config.yaml")
    os.remove("tests/rl_agent/temp_train.py")
    os.remove("tests/rl_agent/temp_evaluate.py")