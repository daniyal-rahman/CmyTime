
import matplotlib.pyplot as plt
import numpy as np

def visualize_weights_over_time(weight_history, title="Weight Sum Over Time"):
    plt.figure(figsize=(10, 6))
    plt.plot(weight_history)
    plt.xlabel("Simulation Step")
    plt.ylabel("Sum of Chemical Weights")
    plt.title(title)
    plt.grid(True)
    plt.savefig("results/figures/weight_sum_over_time.png")
    plt.close()
