# Experiment Log

## Experiment 1: Initial Run

*   **Description:** Initial run of the simulation with the original code.
*   **Observations:** No learning was observed. The `delta_t` values were always zero, and the reward was always negative.

## Experiment 2: `delta_t` Fix

*   **Description:** Corrected the `delta_t` calculation in `src/simulation/engine.py`.
*   **Observations:** `delta_t` values were no longer always zero, but learning was still not effective.

## Experiment 3: Reward and STDP Changes

*   **Description:** Changed the reward calculation to be based on the change in distance to the stimulus and simplified the STDP rule.
*   **Observations:** The simulation ran, but the learning was still not effective.

## Experiment 4: Graded Motor Control

*   **Description:** Changed the motor output to use the `membrane_potential` of the motor neurons.
*   **Observations:** The simulation ran, but the learning was still not effective.

## Experiment 5: Neuron Parameter Tuning and Sensory/Motor Mapping

*   **Description:** Adjusted `LIFNeuron` parameters (`threshold`, `reset_potential`, `initial_potential`) to make neurons more excitable. Implemented a more accurate sensory and motor mapping in `src/simulation/engine.py` using neuron type information from `NeuronTypeOrdered_040903.mat`. Increased `learning_rate`, `a_plus`, and `a_minus` in `configs/default_hparams.yaml`.
*   **Observations:** The simulation runs with stronger weight changes. Visual inspection of `chemoattraction_trajectory.png` and `weight_sum_over_time.png` is needed to assess learning.

## Experiment 6: Increased Neuron Excitability and Input Scaling

*   **Description:** Further decreased `threshold` and increased `initial_potential` in `LIFNeuron`. Increased input current scaling in `engine.py`.
*   **Observations:** Still no significant learning. `delta_w` values remain mostly zero, indicating insufficient neuron firing to drive STDP.

**Next Steps:**

1.  **Further Increase Neuron Excitability:** Continue to adjust `LIFNeuron` parameters (`tau`, `threshold`, `reset_potential`, `initial_potential`) to ensure frequent and varied spiking.
2.  **Increase Input Current Scaling:** Experiment with even higher input current scaling in `engine.py` to ensure sensory neurons are strongly activated.
3.  **Review STDP Parameters:** Re-evaluate `tau_plus` and `tau_minus` in `configs/default_hparams.yaml` to ensure they are appropriate for the new spiking rates. Consider making `a_plus` and `a_minus` even larger.
4.  **Debugging Neuron Activity:** If learning still doesn't occur, consider adding temporary logging within the `LIFNeuron.step` method to confirm that neurons are indeed spiking and that `input_current` values are sufficiently high.
5.  **Initial Weights:** Investigate the initial weight distribution. If weights are too low, it might be difficult for potentiation to occur.
