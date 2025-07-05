
# Current Issues and Proposed Solutions

## 1. Lack of Learning in Chemoattraction Environment

**Issue:** The worm's trajectory in the `ChemoattractionEnv` consistently shows movement only along the X-axis, and the total reward accumulated remains highly negative, indicating that the worm is not learning to approach the stimulus. The weight sum over time also shows an "L-shaped" graph, suggesting rapid depression of weights to their minimum bound.

**Root Cause Analysis:**

*   **STDP Effectiveness:** Debugging logs from `STDP.update` show `delta_t` values consistently at `0.00`, implying that the pre- and post-synaptic neurons are not spiking with the necessary timing differences for effective potentiation or depression. This could be due to:
    *   **Insufficient Neuron Spiking:** Neurons might not be spiking frequently enough, or at all, to generate meaningful `delta_t` values.
    *   **Suboptimal Neuron Parameters:** The `LIFNeuron` parameters (threshold, reset potential, tau) might not be conducive to generating diverse and frequent spiking patterns.
    *   **Ineffective Sensory Input:** The mapping of environmental observations (`gradient_x`, `gradient_y`) to neuron input might not be effectively driving sensory neuron activity.
    *   **Arbitrary Motor Output:** The mapping of motor neuron spikes to environmental actions (turn, move forward) is currently arbitrary and not optimized for the task.
    *   **STDP Parameter Tuning:** The `learning_rate`, `tau_plus`, `tau_minus`, `a_plus`, and `a_minus` parameters in `default_hparams.yaml` might not be appropriately tuned for the current neuron and environment dynamics.
    *   **Reward Modulation:** The current reward modulation in `STDP.update` is a simple linear scaling, which might not be effective enough to guide learning.

**Proposed Solutions & Next Steps:**

1.  **Verify Neuron Spiking (Completed):** Uncommented print statements in `LIFNeuron.step` to confirm if neurons are spiking. (Confirmed that neurons are spiking, but `delta_t` is still 0.00 in STDP updates, indicating a timing issue).

2.  **Refine Neuron Parameters:** Experiment with `LIFNeuron` parameters (`threshold`, `reset_potential`, `tau`) to encourage more varied and frequent spiking, which is crucial for STDP.

3.  **Improve Sensory-Motor Mapping:**
    *   **Sensory Input:** Explore alternative mappings for `gradient_x` and `gradient_y` to sensory neurons. Consider using a larger group of sensory neurons or a more complex non-linear mapping.
    *   **Motor Output:** Develop a more sophisticated mapping from motor neuron activity to `turn_left`, `turn_right`, and `move_forward` actions. This might involve using the *rate* of spiking or the *membrane potential* of motor neurons for graded control, rather than just binary spikes.

4.  **Advanced STDP Implementation:**
    *   **Eligibility Traces:** Implement eligibility traces for STDP. This allows for credit assignment over longer time scales, which is essential for learning with delayed rewards in environments.
    *   **Homeostatic Plasticity:** Introduce homeostatic plasticity mechanisms to prevent weights from collapsing to zero or saturating at maximum values. This can help maintain network stability and responsiveness.
    *   **Parameter Search:** Conduct a systematic hyperparameter search for STDP parameters (`learning_rate`, `tau_plus`, `tau_minus`, `a_plus`, `a_minus`, `reward_strength`) using techniques like grid search or Bayesian optimization.

5.  **Environment Complexity:** While the current environment is simple, ensure that its parameters (`movement_speed`, `rotation_scale`) are well-suited to allow the worm to physically navigate and reach the stimulus given its motor capabilities.

6.  **Debugging STDP (Revisit):** Once spiking is confirmed and sensory-motor mapping is improved, re-examine the `delta_t` values and `delta_w` calculations in `STDP.update` to ensure the STDP rule is being applied correctly and effectively.

## 2. General Code Structure and Modularity

**Issue:** The current implementation, while functional, could benefit from increased modularity and adherence to best practices for larger-scale simulations.

**Proposed Solutions:**

*   **Configuration Management:** Centralize all simulation and environment parameters in configuration files (e.g., YAML) and load them dynamically. (Partially implemented, but can be further refined).
*   **Neuron Model Abstraction:** Ensure the `Neuron` base class and its subclasses (`LIFNeuron`) are flexible enough to easily swap in different neuron models (e.g., Izhikevich neurons) without extensive code changes.
*   **Logging and Checkpointing:** Implement robust logging and checkpointing mechanisms to save simulation state and key metrics at regular intervals. This is crucial for long-running experiments and debugging.
*   **Testing:** Expand unit tests to cover more components and edge cases, especially for the core simulation logic and plasticity rules.
