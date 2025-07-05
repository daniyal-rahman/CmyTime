
# Codebase Overview

This document provides an overview of the files and their functionalities within the `CmyTime` project.

## Project Structure

```
project-root/
├── data/
│   ├── raw/                      # Original, unprocessed connectome data
│   └── processed/                # Processed connectome data (matrices, index maps)
├── configs/
│   ├── default_hparams.yaml      # Default hyperparameters for STDP and other models
│   └── env_params.yaml           # Parameters for the simulation environment
├── src/
│   ├── connectome/
│   │   ├── loader.py             # Handles loading and initial processing of connectome data
│   │   ├── validator.py          # Contains functions for validating connectome data
│   │   └── exporter.py           # Handles exporting processed connectome data
│   ├── models/
│   │   ├── neuron.py             # Base class for neuron models
│   │   ├── lif_neuron.py         # Leaky Integrate-and-Fire (LIF) neuron implementation
│   │   └── plasticity.py         # Implements STDP (Spike-Timing Dependent Plasticity) rules
│   ├── simulation/
│   │   ├── engine.py             # Core simulation loop, handles neuron stepping and plasticity application
│   │   ├── runner.py             # High-level API to run simulation experiments
│   │   └── utils.py              # Utility functions for simulation (e.g., logging)
│   ├── environment/
│   │   ├── base_env.py           # Abstract base class for environments
│   │   └── chemoattraction_env.py# Specific environment for C. elegans chemoattraction task
│   ├── analysis/
│   │   ├── visualize_weights.py  # Script for visualizing weight distributions
│   │   ├── firing_rate_trends.py # Script for analyzing firing rate trends
│   │   └── visualize_weights_over_time.py # Script for visualizing weight sum over time
│   └── main.py                   # Main entry point for running simulations
├── scripts/
│   ├── preprocess.sh             # Shell script for data preprocessing (placeholder)
│   ├── run_experiments.sh        # Shell script for running multiple experiments (placeholder)
│   └── process_raw_data.py       # Python script to process raw .mat connectome data
├── tests/
│   ├── test_loader.py            # Unit tests for the connectome loader
│   ├── test_neuron_models.py     # Unit tests for neuron models
│   ├── test_env.py               # Unit tests for the candy button environment
│   └── test_environment.py       # Unit tests for the chemoattraction environment
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

*   **`data/raw/`**: This directory is intended to store the original, unprocessed C. elegans connectome data. Currently, it contains MATLAB `.mat` files (`ConnOrdered_040903.mat`, `NeuronTypeOrdered_040903.mat`) which are the raw data sources.
*   **`data/processed/`**: This directory stores the connectome data after it has been processed into a more usable format for the simulation. It contains:
    *   `adj_matrices.npz`: A NumPy compressed archive containing the chemical and electrical adjacency matrices of the connectome.
    *   `neuron_index_map.json`: A JSON file mapping neuron names to their integer indices used in the adjacency matrices.

### `configs/`

*   **`configs/default_hparams.yaml`**: A YAML file defining the default hyperparameters for the STDP rule, such as `learning_rate`, `tau_plus`, `tau_minus`, `a_plus`, `a_minus`, and weight clipping bounds (`w_min`, `w_max`).
*   **`configs/env_params.yaml`**: A YAML file defining parameters for the simulation environment, including `stimulus_location`, `initial_position`, `movement_speed`, and `rotation_scale` for the `ChemoattractionEnv`.

### `src/connectome/`

*   **`src/connectome/loader.py`**:
    *   **Purpose**: Responsible for loading the raw connectome data from the `.mat` files and transforming it into sparse adjacency matrices and a neuron index mapping.
    *   **Key Function**: `load_connectome(path)`: Reads `adj_matrices.npz` and `neuron_index_map.json` from the specified path and returns the chemical adjacency matrix, electrical adjacency matrix, and the neuron-to-index mapping.

*   **`src/connectome/validator.py`**:
    *   **Purpose**: Provides functions to perform basic sanity checks on the loaded connectome data.
    *   **Key Function**: `validate_connectome(adj_matrix)`: Asserts that the input adjacency matrix is square and does not contain NaN values.

*   **`src/connectome/exporter.py`**:
    *   **Purpose**: Handles the saving of processed connectome data to `.npz` files.
    *   **Key Function**: `export_to_npz(adj_chemical, adj_electrical, neuron_index_map, path)`: Saves the chemical and electrical adjacency matrices and the neuron index map to `.npz` files in the specified directory.

### `src/models/`

*   **`src/models/neuron.py`**:
    *   **Purpose**: Defines the abstract base class for all neuron models in the simulation.
    *   **Key Class**: `Neuron`: Provides a basic structure with `membrane_potential` and an abstract `step` method that subclasses must implement.

*   **`src/models/lif_neuron.py`**:
    *   **Purpose**: Implements a Leaky Integrate-and-Fire (LIF) neuron model, inheriting from `Neuron`.
    *   **Key Class**: `LIFNeuron`: Simulates the dynamics of a LIF neuron, including membrane potential integration, spiking, and resetting. It also tracks the `last_spike_time` for STDP calculations.

*   **`src/models/plasticity.py`**:
    *   **Purpose**: Implements the Spike-Timing Dependent Plasticity (STDP) rule, which modifies synaptic weights based on the relative timing of pre- and post-synaptic spikes.
    *   **Key Class**: `STDP`: Contains the `update` method that calculates `delta_w` based on `delta_t` (time difference between pre- and post-synynaptic spikes) and applies weight changes, including reward modulation and weight clipping.

### `src/simulation/`

*   **`src/simulation/engine.py`**:
    *   **Purpose**: Contains the core simulation logic, responsible for advancing the simulation by one time step. It integrates neuron dynamics, applies plasticity rules, and interacts with the environment.
    *   **Key Class**: `Engine`: Manages the `connectome`, `neurons`, and `plasticity_rule`. Its `step` method handles sensory input mapping, neuron spiking, motor output mapping, environment interaction, and STDP updates.

*   **`src/simulation/runner.py`**:
    *   **Purpose**: Provides a high-level interface to run simulation experiments for a specified number of steps. It collects and stores simulation data like rewards, positions, and weight history.
    *   **Key Class**: `Runner`: Orchestrates the simulation by repeatedly calling the `engine.step()` method and recording relevant data.

*   **`src/simulation/utils.py`**:
    *   **Purpose**: Contains general utility functions that support the simulation, such as logging setup.
    *   **Key Function**: `setup_logging()`: Configures basic logging for the application.

### `src/environment/`

*   **`src/environment/base_env.py`**:
    *   **Purpose**: Defines the abstract base class for all simulation environments.
    *   **Key Class**: `BaseEnv`: Specifies the required methods for an environment, such as `reset()`, `step()`, `get_observation()`, and `get_reward()`.

*   **`src/environment/chemoattraction_env.py`**:
    *   **Purpose**: Implements a specific environment where the C. elegans worm attempts to navigate towards a chemical stimulus.
    *   **Key Class**: `ChemoattractionEnv`: Manages the worm's position, calculates distance and gradient to the stimulus, and provides rewards based on proximity. It interprets motor actions as changes in angle and forward movement.

### `src/analysis/`

*   **`src/analysis/visualize_weights.py`**:
    *   **Purpose**: Provides functionality to visualize the distribution of synaptic weights.
    *   **Key Function**: `visualize_weights(weights, trial_num)`: Generates and saves a histogram of the given weights.

*   **`src/analysis/firing_rate_trends.py`**:
    *   **Purpose**: Provides functionality to plot learning curves or firing rate trends over time.
    *   **Key Function**: `plot_learning_curve(rewards)`: Generates and saves a plot of accumulated rewards over simulation steps.

*   **`src/analysis/visualize_weights_over_time.py`**:
    *   **Purpose**: Provides functionality to visualize how the sum of weights changes throughout a simulation.
    *   **Key Function**: `visualize_weights_over_time(weight_history)`: Generates and saves a plot of the total sum of chemical weights at each simulation step.

### `src/`

*   **`src/main.py`**:
    *   **Purpose**: The main entry point for the simulation. It loads configurations, initializes the network, environment, and simulation components, runs the simulation, and triggers analysis and visualization.
    *   **Key Function**: `main()`: Orchestrates the entire simulation pipeline.

### `scripts/`

*   **`scripts/preprocess.sh`**: A placeholder shell script for data preprocessing steps. In a full setup, this might handle downloading raw data or running initial conversion scripts.
*   **`scripts/run_experiments.sh`**: A placeholder shell script for launching multiple simulation runs, potentially with different hyperparameters.
*   **`scripts/process_raw_data.py`**:
    *   **Purpose**: A Python script responsible for loading the raw MATLAB `.mat` connectome files and converting them into a more efficient and Python-friendly format (`.npz` and `.json`) for use in the simulation.
    *   **Key Function**: `process_raw_data()`: Reads the `.mat` files, extracts adjacency matrices and neuron labels, performs minor data cleaning (e.g., removing self-loops), and saves the processed data.

### `tests/`

*   **`tests/test_loader.py`**: Contains unit tests to verify the correct loading and initial processing of connectome data by `src/connectome/loader.py`.
*   **`tests/test_neuron_models.py`**: Contains unit tests to verify the basic functionality and spiking behavior of the `LIFNeuron` model.
*   **`tests/test_env.py`**: Contains unit tests for the `CandyButtonEnv` (from the original roadmap, though not currently used in `main.py`).
*   **`tests/test_environment.py`**: Contains unit tests for the `ChemoattractionEnv`, verifying its reset and step functionalities, and reward calculation.

### Other Files

*   **`Dockerfile`**: Defines the Docker image for the project, ensuring a reproducible and isolated environment with all necessary dependencies.
*   **`requirements.txt`**: Lists all Python package dependencies required to run the project.
*   **`pytest.ini`**: Configuration file for `pytest`, specifying how tests should be discovered and run.
*   **`issue_description.md`**: A separate document (as requested) detailing the current known issues with the simulation's learning capabilities and outlining potential debugging and solution strategies.
