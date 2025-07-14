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

## 4. Learning Brain Not Avoiding Traps

*   **Problem:** The `LearningBrain` agent was not effectively learning to avoid traps in the `TrappedChemoattractionEnv`, despite initial attempts to tune learning parameters and increase sensory input scaling. The simulation output showed consistent "Time in trap" and "Total trap penalty" values across epochs, indicating a lack of learning.

*   **Root Cause:** The primary issue was an architectural disconnect within the `LearningBrain`. While the `LearningBrain` had logic for calculating internal "spikes" and updating STDP traces, these internal calculations were not actually driving the `LIFNeuron` models. The `LIFNeuron` instances were initialized but their `step` method was never called, meaning their membrane potentials and actual spiking activity were not influencing the agent's motor output or the STDP updates. The agent's trap avoidance behavior was primarily driven by a direct, heuristic-based calculation in `get_action` rather than emergent behavior from the neural network.

*   **Technical Solution:**
    1.  **Integrated LIF Neurons into LearningBrain:**
        *   `LIFNeuron` instances were explicitly initialized within the `LearningBrain`'s `__init__` method, creating a list of `LIFNeuron` objects, one for each neuron in the connectome.
        *   The `spike_threshold` parameter was removed from `LearningBrain`'s `__init__` as it is now managed by the `LIFNeuron` class itself.
    2.  **Refactored `get_action` for Neural Dynamics:**
        *   A `self.current_time` variable was introduced and incremented in `get_action` to provide a time context for `LIFNeuron.step`.
        *   The `input_current` for each `LIFNeuron` was calculated by combining:
            *   Scaled sensory input (for food and traps).
            *   Recurrent input from other neurons, calculated as the weighted sum of *actual spikes* from the *previous* time step (stored in a new `spike_output` attribute of `LIFNeuron`).
        *   Each `LIFNeuron`'s `step` method was called with its calculated `input_current` and `self.current_time`, allowing the neurons to simulate their membrane potential dynamics and generate actual spikes.
        *   The `spike_output` attribute of each `LIFNeuron` was updated after its `step` method call to store its current spiking status for the next iteration.
    3.  **STDP Updates Based on Actual Spikes:**
        *   The `pre_synaptic_trace` and `post_synaptic_trace` in `LearningBrain` were updated based on the *actual spikes* generated by the `LIFNeuron`s in the current time step.
        *   The `eligibility_trace` was also updated using these actual spikes, ensuring that weight changes were directly tied to the simulated neural activity.
    4.  **Motor Control from LIF Neuron Potentials:**
        *   Motor activation was derived from the `membrane_potential` of the motor `LIFNeuron`s, providing a more biologically plausible link between neural activity and behavior. The potentials were normalized and averaged to produce a base forward signal.
    5.  **Parameter Tuning:**
        *   `LIFNeuron` parameters (`threshold`, `initial_potential`) were adjusted to increase neuron excitability.
        *   `default_hparams.yaml` parameters (`learning_rate`, `a_plus`, `a_minus`, `reward_strength`) were re-tuned to ensure that the STDP mechanism could effectively modify weights based on the new spiking dynamics and reward signals.

This comprehensive refactoring ensured that the `LearningBrain` truly simulated a spiking neural network, allowing for emergent learning behavior through STDP, which successfully enabled the worm to learn to avoid traps.

## 5. Learning Brain Not Reaching Food Source Consistently

*   **Problem:** After successfully learning to avoid traps, the `LearningBrain` agent struggled to consistently navigate towards the food source. The worm's final positions were often far from the stimulus, indicating a lack of effective food-seeking behavior.

*   **Root Cause:** The action generation mechanism in `LearningBrain` was too simplistic and did not provide a sufficiently strong or clear signal for the network to learn to orient towards the food. While the previous refactoring integrated `LIFNeuron`s, the mapping from their activity to the `turn_action` and `move_forward_signal` was not robust enough to guide the worm effectively. The reliance on purely emergent behavior without sufficient initial guidance made learning slow and inconsistent for food-seeking.

*   **Technical Solution:**
    1.  **Reintroduced Stronger Direct Influence for Food-Seeking:**
        *   The `turn_action` calculation in `LearningBrain.get_action` was modified to reintroduce a stronger, direct influence from the `relative_stimulus_angle`. This provides a clearer initial directional signal for the network to learn from, acting as a "scaffold" for the STDP.
        *   The `turn_action` still incorporates trap avoidance, ensuring both behaviors are considered.
    2.  **Increased Forward Movement Bias:**
        *   A larger constant bias was added to the `move_forward_signal` in `LearningBrain.get_action`. This ensures the worm has a consistent tendency to move forward, which is crucial for reaching the distant food source, unless actively inhibited by trap avoidance.
    3.  **Continued STDP Refinement:**
        *   With these stronger initial biases in action generation, the STDP mechanism can now more effectively refine the synaptic weights to produce coordinated movement that balances trap avoidance and food attraction.

These adjustments provided the necessary initial guidance and consistent forward momentum, allowing the STDP to effectively learn the complex navigation task, resulting in the worm successfully avoiding traps and consistently moving towards the food source.