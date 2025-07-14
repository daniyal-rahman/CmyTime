# Codebase Overview

This document provides an overview of the files and their functionalities within the `CmyTime` project, reflecting the recent refactoring to a modular, agent-centric architecture.

## Project Structure

```
project-root/
├── data/
│   ├── raw/                      # Original, unprocessed connectome data
│   └── processed/                # Processed connectome data (matrices, index maps)
├── configs/
│   ├── default_hparams.yaml      # Default hyperparameters
│   └── env_params.yaml           # Parameters for the simulation environment
├── src/
│   ├── brains/
│   │   ├── base_brain.py         # Abstract base class for all brain models
│   │   ├── static_brain.py       # Implements a static brain model using connectome data
│   │   └── learning_brain.py     # Implements a brain with learning capabilities
│   ├── core/
│   │   ├── agent.py              # Encapsulates the brain and body of the simulated organism
│   │   ├── body.py               # Defines the physical state and properties of the agent
│   │   ├── base_env.py           # Abstract base class for environments (Gymnasium-compliant)
│   │   └── simulator.py          # Orchestrates the simulation loop
│   ├── environments/
│   │   ├── chemoattraction_env.py# Specific environment for C. elegans chemoattraction task
│   │   └── trapped_chemoattraction_env.py # Environment with traps to avoid
│   └── main.py                   # Main entry point for running simulations
├── scripts/                      # Utility scripts
├── tests/                        # Unit tests for various components
├── Dockerfile                    # Dockerfile for reproducible environment setup
├── requirements.txt              # Python dependency list
├── README.md                     # Project README
├── pytest.ini                    # Pytest configuration
└── docs/
    ├── issue_description.md      # Document detailing current issues and proposed fixes
    ├── code_overview.md          # This document
    └── troubleshooting.md        # Document detailing troubleshooting and resolutions
```

## File Descriptions

### `data/`

*   **`data/raw/`**: Contains the C. elegans connectome data from the OpenWorm project.

### `configs/`

*   **`configs/default_hparams.yaml`**: Hyperparameters for the simulation.
*   **`configs/env_params.yaml`**: Environment parameters.

### `src/brains/`

*   **`src/brains/base_brain.py`**: Defines the `BaseBrain` abstract class.
*   **`src/brains/static_brain.py`**: Implements a static brain model that can navigate towards a stimulus.
*   **`src/brains/learning_brain.py`**: Implements a brain with learning capabilities, now fully integrating `LIFNeuron` models for neural dynamics and STDP-based weight updates.

### `src/core/`

*   **`src/core/agent.py`**: The `Agent` class, which combines a brain and a body.
*   **`src/core/body.py`**: The `Body` class, which holds the agent's physical state.
*   **`src/core/base_env.py`**: Defines the `BaseEnv` abstract class, compliant with the Gymnasium API.
*   **`src/core/simulator.py`**: The `Simulator` class, which manages the simulation loop and calls the brain's learning rule.

### `src/environments/`

*   **`src/environments/chemoattraction_env.py`**: A simple environment with a single food source.
*   **`src/environments/trapped_chemoattraction_env.py`**: A more complex environment with a food source and traps to avoid.

### `src/main.py`

*   **`src/main.py`**: The main entry point for running simulations. It uses `argparse` to allow for selecting different brains and environments.
