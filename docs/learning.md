# Learning Brain and Trapped Environment

This document describes the `LearningBrain` and the `TrappedChemoattractionEnv`, which were created to add learning capabilities to the simulation.

## `LearningBrain`

The `LearningBrain` is a brain model that has the ability to modify its connectome based on experience. It inherits from the `BaseBrain` and uses the same connectome data as the `StaticBrain`.

### Learning Mechanism

The `LearningBrain` uses a simple reinforcement learning rule to update its weights. The rule is as follows:

1.  After each action, the brain receives a reward from the environment.
2.  If the reward is positive, the connections between the sensory neurons that were active and the motor neurons that were active are strengthened.
3.  If the reward is negative, those same connections are weakened.

This allows the agent to learn which actions are beneficial and which are detrimental in a given situation.

**Note:** The current implementation of the learning rule is still under development and is not yet effective at producing robust learning behavior.

## `TrappedChemoattractionEnv`

The `TrappedChemoattractionEnv` is a more complex environment designed to test the `LearningBrain`. It contains:

*   **A food source:** This provides a positive reward to the agent.
*   **Traps:** These are areas that provide a negative reward to the agent.

The agent must learn to navigate to the food source while avoiding the traps.

### Observation Space

The observation space of this environment is an extension of the `ChemoattractionEnv`'s observation space. It includes the following additional information:

*   `distance_to_nearest_trap`: The distance to the nearest trap.
*   `relative_trap_angle`: The angle to the nearest trap, relative to the agent's current orientation.

This information allows the agent to perceive the traps and learn to avoid them.
