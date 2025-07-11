# Codebase Overview

This document provides an overview of the files and their functionalities within the `CmyTime` project, reflecting the recent refactoring to a modular, agent-centric architecture.

## Project Structure

```
project-root/
├── data/
│   ├── raw/                      # Original, unprocessed connectome data
│   └── processed/                # Processed connectome data (matrices, index maps)
├── configs/
│   ├── default_hparams.yaml      # Default hyperparameters (currently less relevant for StaticBrain)
│   └── env_params.yaml           # Parameters for the simulation environment
├── src/
│   ├── brains/
│   │   ├── base_brain.py         # Abstract base class for all brain models
│   │   └── static_brain.py       # Implements a static brain model using connectome data
│   ├── core/
│   │   ├── agent.py              # Encapsulates the brain and body of the simulated organism
│   │   ├── body.py               # Defines the physical state and properties of the agent
│   │   ├── base_env.py           # Abstract base class for environments (Gymnasium-compliant)
│   │   └── simulator.py          # Orchestrates the simulation loop
│   ├── environments/
│   │   └── chemoattraction_env.py# Specific environment for C. elegans chemoattraction task (Gymnasium-compliant)
│   └── main.py                   # Main entry point for running simulations
├── scripts/                      # Utility scripts (e.g., data processing)
├── tests/                        # Unit tests for various components
├── Dockerfile                    # Dockerfile for reproducible environment setup
├── requirements.txt              # Python dependency list
├── README.md                     # Project README
├── pytest.ini                    # Pytest configuration
└── docs/
    ├── issue_description.md      # Document detailing current issues and proposed fixes
    └── code_overview.md          # This document
```

## File Descriptions

### `data/`

*   **`data/raw/`**: Contains raw C. elegans connectome data, including `NeuronConnectFormatted.xlsx` (for connection weights) and `NeuronTypeOrdered_040903.mat` (for neuron types).
*   **`data/processed/`**: Intended for processed data (currently not actively used by `StaticBrain`).

### `configs/`

*   **`configs/default_hparams.yaml`**: Hyperparameters (less relevant for the current `StaticBrain` but important for future learning models).
*   **`configs/env_params.yaml`**: Environment parameters.

### `src/brains/`

*   **`src/brains/base_brain.py`**: Defines the `BaseBrain` abstract class with a `get_action` method that all brain models must implement.
*   **`src/brains/static_brain.py`**: Implements `StaticBrain`. It loads connectome data and heuristically categorizes neurons into sensory, motor, and interneurons based on their connectivity patterns. It then uses a simplified model to map sensory input to motor actions.

### `src/core/`

*   **`src/core/agent.py`**: The `Agent` class combines a `brain` (e.g., `StaticBrain`) and a `body` to represent the simulated organism. Its `act` method calls the brain's `get_action` method.
*   **`src/core/body.py`**: The `Body` class holds the agent's physical state (position, angle) and will be responsible for updating these based on actions from the brain.
*   **`src/core/base_env.py`**: Defines the `BaseEnv` abstract class, which now inherits from `gymnasium.Env`, providing a standardized interface for environments.
*   **`src/core/simulator.py`**: The `Simulator` class manages the overall simulation loop, interacting with the `Agent` and `Environment` to advance the simulation step by step. It also collects trajectory data.

### `src/environments/`

*   **`src/environments/chemoattraction_env.py`**: Implements the `ChemoattractionEnv`, a specific environment where the agent navigates towards a chemical stimulus. It provides observations (including `gradient_x` and `gradient_y`) and processes actions according to the Gymnasium API.

### `src/main.py`

*   **`src/main.py`**: The main entry point. It initializes the environment, brain, body, agent, and simulator, then runs the simulation and generates the trajectory plot.
