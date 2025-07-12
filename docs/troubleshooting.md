# Troubleshooting and Resolutions

This document outlines the key issues encountered during the development and debugging of the C. elegans simulation, along with the solutions that were implemented.

## 1. Initial Setup and Data Integration

*   **Problem:** The project contained multiple C. elegans connectome datasets (`VarshneyEtAl2011`, `CElegansNeuroML`, and `celegans_connectome.json`), and the codebase referenced a dataset that was not well-suited for simulation. The data loading scripts were either obsolete or pointed to the wrong files.

*   **Solution:**
    1.  **Dataset Evaluation:** The available datasets were evaluated. The `CElegansNeuroML` dataset from the OpenWorm project was identified as the most suitable for simulation due to its focus on a dynamic, simulatable model.
    2.  **Data Cleanup:** The `VarshneyEtAl2011` and `celegans_connectome.json` datasets were removed from the `data/raw` directory to avoid confusion.
    3.  **Code Refactoring:** All references to the deleted datasets were removed from the codebase. The `src/main.py` file was updated to point to the correct connectome and neuron type files within the `CElegansNeuroML` directory. The obsolete `scripts/process_raw_data.py` script was also deleted.

## 2. Agent Movement and Navigation

*   **Problem:** The agent was not moving towards the stimulus. Initial runs showed the agent moving in the wrong direction, away from the food source. This was attributed to several factors, including incorrect neuron categorization and overly simplistic action generation logic.

*   **Solution:**
    1.  **Neuron Categorization:** The `_categorize_neurons` method in `src/brains/static_brain.py` was completely rewritten. The initial implementation used a faulty method to read the neuron type data and relied on a very small, hardcoded list of motor neurons. The new implementation:
        *   Correctly reads the `CElegansNeuronTables.xls` file using the `pandas` library.
        *   Attempts to identify sensory and motor neurons based on a 'Function' column in the 'NeuronsToMuscle' sheet.
        *   Includes a robust fallback mechanism that uses a more comprehensive list of motor neurons and a heuristic to identify sensory neurons from the main connectome data.
    2.  **Action Generation:** The `get_action` method in `src/brains/static_brain.py` was refined to better translate motor neuron activation into agent movement. The initial implementation was too simplistic and did not produce goal-directed behavior.

*   **Problem:** After the initial fixes, the agent was able to orient itself towards the stimulus but would then spiral away from it. This indicated that the turning logic was too aggressive and not properly scaled.

*   **Solution:**
    1.  **Proportional Control:** The `get_action` method was rewritten to use a proportional control mechanism for turning. The turning action is now directly proportional to the `relative_stimulus_angle`, meaning the agent turns more sharply when it is farther off-course.
    2.  **Speed Modulation:** The forward speed is now modulated based on the agent's alignment with the stimulus. The agent moves faster when it is pointing directly at the food source and slows down when it needs to make a turn. This results in more controlled and deliberate movement.

## 3. Dependency Issues

*   **Problem:** The simulation would crash due to a `ModuleNotFoundError` for the `xlrd` library.

*   **Solution:** The `xlrd` library, which is required by `pandas` to read `.xls` files, was not listed in the project's dependencies. The issue was resolved by installing `xlrd` using `pip`.
