# Current Issues and Proposed Solutions

## 1. Agent Movement and Navigation

**Status:** **Resolved.** The agent's movement model has been fundamentally refactored to be more biologically plausible, addressing the limitations of the previous heuristic-based approach.

**Previous Problem:** The mapping from motor neuron activation to environment actions was a simplified heuristic (`turn_left`, `turn_right`, `move_forward`). This did not accurately represent the locomotion of C. elegans, which relies on the coordinated contraction of dorsal and ventral muscles. This abstraction limited the potential for emergent, realistic behavior.

**Solution: Biologically-Plausible Movement Model**
*   **Dorsal/Ventral Control:** The model now explicitly simulates the differential activation of dorsal and ventral muscle groups.
    *   Motor neurons in the connectome have been categorized into `dorsal_motor` (DA, DB, AS) and `ventral_motor` (VA, VB, VD, VC) groups.
    *   The `get_action` method in the brain now computes the total activation for these two groups.
*   **New Action Space:** The brain no longer returns abstract turn/move commands. Instead, the action is a 2-element array: `[dorsal_activation, ventral_activation]`.
*   **Environment Physics:** The environments now interpret this action to produce movement:
    *   **Turning:** The change in angle is proportional to the *difference* in activations (`dorsal - ventral`), simulating how uneven muscle contraction causes the worm to turn.
    *   **Forward Movement:** The forward speed is proportional to the *sum* of activations (`dorsal + ventral`), simulating how overall muscle activity drives locomotion.

This new model provides a much more accurate foundation for studying emergent behavior, as the agent must now learn to control its "muscles" in a coordinated way to navigate its environment.

## 2. General Code Structure and Modularity

**Status:** The codebase has been successfully refactored to a modular, agent-centric architecture compatible with the Gymnasium API. Key components (`Agent`, `Body`, `StaticBrain`, `ChemoattractionEnv`, `Simulator`) are in place and integrated. The previous `src/simulation` and `src/environment` directories have been removed.

**Future Work:** Continue to refine modularity, expand test coverage, and explore integration with other Gymnasium environments and reinforcement learning frameworks.