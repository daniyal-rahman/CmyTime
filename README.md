# C. Elegans Connectome Simulation

This project aims to simulate the C. elegans connectome with STDP (Spike-Timing Dependent Plasticity) learning in a toy environment. The goal is to eventually scale this up to the fruit fly connectome and simulate reactions to an environment.

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
3.  **Process raw data:**
    ```bash
    python scripts/process_raw_data.py
    ```

## Usage

To run the simulation:

```bash
python -m src.main
```

This will run the simulation, apply STDP, and generate visualizations in the `results/figures/` directory.

## Current Issues and Future Work

For a detailed description of current issues and proposed solutions, please refer to `docs/issue_description.md`.