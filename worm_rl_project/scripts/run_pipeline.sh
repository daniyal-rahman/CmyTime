#!/bin/bash

# Change to the project's root directory
cd "$(dirname "$0")/../.."

# Call generate_model.sh
echo "Generating models..."
worm_rl_project/scripts/generate_model.sh

# Trigger train.py
echo "Training the RL agent..."
python -m worm_rl_project.rl_agent.train

# Trigger evaluate.py
echo "Evaluating the RL agent..."
python -m worm_rl_project.rl_agent.evaluate

echo "Pipeline finished."
echo "Saved models and logs can be found in the 'models' directory."

