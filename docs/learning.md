# Learning Brain and Trapped Environment

This document describes the `LearningBrain` and the `TrappedChemoattractionEnv`, which are used to test the agent's ability to learn and adapt.

## `LearningBrain`

The `LearningBrain` is a brain model that can modify its connectome based on experience. It uses Leaky Integrate-and-Fire (LIF) neurons to simulate neural dynamics and a Spike-Timing-Dependent Plasticity (STDP) rule to update its synaptic weights.

### Learning Mechanism

The `LearningBrain` must learn to control its movement by modulating the activations of its dorsal and ventral motor neurons.

1.  **Action Generation**: At each step, the brain receives sensory information (e.g., food gradient, trap location). This input propagates through the spiking neural network, resulting in membrane potential changes in the motor neurons. The average membrane potential of the dorsal and ventral motor neuron groups is used to generate the action `[dorsal_activation, ventral_activation]`.
2.  **Reward-Modulated STDP**: After the action is taken, the environment provides a reward signal. This reward modulates the STDP learning rule.
    *   If the reward is positive (e.g., moving closer to food), synaptic connections that were recently active (as tracked by an eligibility trace) are strengthened.
    *   If the reward is negative (e.g., moving into a trap), those same connections are weakened.

This process allows the agent to learn which patterns of neural activity lead to beneficial outcomes, effectively learning how to "drive" its own body to find food and avoid traps.

## `TrappedChemoattractionEnv`

The `TrappedChemoattractionEnv` is a more complex environment designed to test the `LearningBrain`. It contains:

*   **A food source:** This provides a positive reward to the agent.
*   **Traps:** These are areas that provide a large negative reward to the agent.

The agent must learn to navigate to the food source while avoiding the traps by coordinating its dorsal and ventral "muscles".

### Observation Space

The observation space includes information about the food stimulus (distance, angle, gradient) and the nearest trap (distance, angle), allowing the agent to perceive the key elements of its environment.
