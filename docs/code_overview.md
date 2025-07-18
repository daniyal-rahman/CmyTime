# Codebase Overview

This document provides an overview of the files and their functionalities within the `CmyTime` project, reflecting the recent refactoring to a modular, agent-centric architecture with a biologically-plausible movement model.

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
├─�� Dockerfile                    # Dockerfile for reproducible environment setup
├── requirements.txt              # Python dependency list
├── README.md                     # Project README
├── pytest.ini                    # Pytest configuration
└── docs/
    ├── issue_description.md      # Document detailing current issues and proposed fixes
    ├── code_overview.md          # This document
    └── troubleshooting.md        # Document detailing troubleshooting and resolutions
```

## File Descriptions

### `src/brains/`

The brain components are responsible for processing observations and generating actions. They now model the C. elegans nervous system by mapping sensory input to motor neuron activations.

*   **`src/brains/base_brain.py`**: Defines the `BaseBrain` abstract class.
*   **`src/brains/static_brain.py`**: Implements a brain with a fixed-synapse connectome. It generates actions by propagating sensory input through the network to dorsal and ventral motor neuron groups.
*   **`src/brains/learning_brain.py`**: Implements a brain with plastic synapses. It uses `LIFNeuron` models for neural dynamics and an STDP-based learning rule to modify weights. Actions are generated from the membrane potentials of the dorsal and ventral motor neurons.

### `src/environments/`

The environment components define the simulation world and its physics. The action space for all environments has been refactored to be more biologically plausible.

*   **Action Space**: The action is a 2-element array `[dorsal_activation, ventral_activation]`, where each value is between 0 and 1.
*   **Movement Logic**:
    *   **Turning**: Proportional to the *difference* between dorsal and ventral activations (`dorsal - ventral`).
    *   **Forward Movement**: Proportional to the *sum* of dorsal and ventral activations (`dorsal + ventral`).

*   **`src/environments/chemoattraction_env.py`**: A simple environment with a single food source.
*   **`src/environments/trapped_chemoattraction_env.py`**: A more complex environment with a food source and traps to avoid.
