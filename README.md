# C. Elegans Connectome Simulation

This project aims to simulate the C. elegans connectome in a modular and extensible environment. The goal is to eventually scale this up to different animal models and simulate various behaviors.

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

To run the simulation:

```bash
python -m src.main
```

This will run the simulation and generate a visualization of the agent's trajectory in the `results/figures/` directory.

## Project Structure and Current Status

This project is currently undergoing a refactoring to a modular, agent-centric architecture, compatible with the Gymnasium API. The core components are:

*   **`src/core/`**: Contains fundamental simulation components like `Agent`, `Body`, and `Simulator`.
*   **`src/brains/`**: Houses different brain models, starting with `StaticBrain` which uses a pre-defined connectome.
*   **`src/environments/`**: Contains environment implementations, currently `ChemoattractionEnv` which follows the Gymnasium API.

The `StaticBrain` is currently implemented to use the C. elegans connectome data to generate actions. However, the agent is not yet successfully navigating towards the food source. Debugging is ongoing to ensure correct sensory input processing and motor activation.

For a detailed overview of the codebase and a guide to debugging the current issue, please refer to `docs/code_overview.md`.
