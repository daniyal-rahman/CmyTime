
import matplotlib.pyplot as plt

def visualize_weights(weights, trial_num):
    plt.hist(weights.flatten(), bins=50)
    plt.title(f"Weight Distribution - Trial {trial_num}")
    plt.xlabel("Weight")
    plt.ylabel("Frequency")
    plt.savefig(f"results/figures/weight_hist_trial{trial_num}.png")
