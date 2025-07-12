# C. Elegans Connectome Simulation

This project aims to simulate the C. elegans connectome in a modular and extensible environment. The goal is to eventually scale this up to different animal models and learning models and simulate various behaviors.

## Setup

1.  **Clone the repository:**
    ```bash
    git clone <repository_url>
    cd CmyTime
    ```
2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

## Usage

To run the simulation, use the `src/main.py` script with the following command-line options:

```bash
python -m src.main --brain <brain_type> --env <env_type> --epochs <num_epochs> --steps <num_steps>
```

*   `--brain`: The type of brain to use. Options are `static` (default) and `learning`.
*   `--env`: The environment to run the simulation in. Options are `chemoattraction` (default) and `trapped`.
*   `--epochs`: The number of epochs to run the simulation for. Default is `1`.
*   `--steps`: The number of steps per epoch. Default is `2000`.

For example, to run the simulation with the learning brain in the trapped environment for 10 epochs:
```bash
python -m src.main --brain learning --env trapped --epochs 10
```

This will run the simulation and generate a visualization of the agent's trajectory in the `results/figures/` directory.

## Project Structure and Current Status

This project is currently undergoing a refactoring to a modular, agent-centric architecture, compatible with the Gymnasium API. The core components are:

*   **`src/core/`**: Contains fundamental simulation components like `Agent`, `Body`, and `Simulator`.
*   **`src/brains/`**: Houses different brain models, including a `StaticBrain` and a `LearningBrain`.
*   **`src/environments/`**: Contains environment implementations, including a simple `ChemoattractionEnv` and a more complex `TrappedChemoattractionEnv`.

The `StaticBrain` is able to successfully navigate to the food source in the `ChemoattractionEnv`. The `LearningBrain` is still under development and is not yet able to learn effectively.

For a detailed overview of the codebase and a guide to debugging the current issue, please refer to `docs/code_overview.md`.
