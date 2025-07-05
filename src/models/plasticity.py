import numpy as np
import scipy.sparse as sp

class STDP:
    def __init__(self, learning_rate, tau_plus, tau_minus, a_plus, a_minus, w_min=0.0, w_max=1.0, reward_threshold=0.0, reward_strength=0.1):
        self.learning_rate = learning_rate
        self.tau_plus = tau_plus
        self.tau_minus = tau_minus
        self.a_plus = a_plus
        self.a_minus = a_minus
        self.w_min = w_min
        self.w_max = w_max
        self.reward_threshold = reward_threshold
        self.reward_strength = reward_strength

    def update(self, weights, pre_spike_times, post_spike_times, current_time, reward=0):
        """Applies STDP updates to the weights based on spike timings, modulated by reward."""
        weights_lil = weights.tolil()
        rows, cols = weights.nonzero()

        # Dynamic adjustment of A_plus and A_minus based on reward
        reward_factor = 1.0 + reward * 0.0001 # Small scaling factor for reward
        current_a_plus = self.a_plus * reward_factor
        current_a_minus = self.a_minus * reward_factor

        updated_connections_count = 0

        for r, c in zip(rows, cols):
            t_pre = pre_spike_times[r]
            t_post = post_spike_times[c]

            # Only apply STDP if both pre- and post-synaptic neurons have spiked
            if t_pre != -np.inf and t_post != -np.inf:
                delta_t = t_post - t_pre

                delta_w = 0
                if delta_t > 0:  # Potentiation
                    delta_w = current_a_plus * np.exp(-delta_t / self.tau_plus)
                elif delta_t < 0: # Depression
                    delta_w = -current_a_minus * np.exp(delta_t / self.tau_minus)

                # Reward modulation
                if reward > self.reward_threshold:
                    # Positive reward enhances potentiation, reduces depression
                    if delta_w > 0:
                        delta_w *= (1 + self.reward_strength * reward)
                    else:
                        delta_w *= (1 - self.reward_strength * reward) # Reduce depression
                elif reward < self.reward_threshold:
                    # Negative reward enhances depression, reduces potentiation
                    if delta_w < 0:
                        delta_w *= (1 + self.reward_strength * abs(reward))
                    else:
                        delta_w *= (1 - self.reward_strength * abs(reward)) # Reduce potentiation

                new_weight = weights_lil[r, c] + self.learning_rate * delta_w
                weights_lil[r, c] = np.clip(new_weight, self.w_min, self.w_max) # Apply clipping
                updated_connections_count += 1

                # Log for any updated connection (first 10 updates per step)
                if updated_connections_count <= 10:
                    print(f"STDP Update - Time: {current_time}, Conn: ({r},{c}), delta_t: {delta_t:.2f}, delta_w: {delta_w:.4f}, Old Weight: {weights_lil[r,c]:.4f}, New Weight: {new_weight:.4f}")

        if updated_connections_count > 0:
            print(f"Total STDP updates this step: {updated_connections_count}")

        return weights_lil.tocsr()