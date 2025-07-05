
import matplotlib.pyplot as plt
import numpy as np

def visualize_chemoattraction(positions_history, stimulus_location, title="Worm Trajectory in Chemoattraction Environment"):
    positions_history = np.array(positions_history)
    plt.figure(figsize=(8, 8))
    plt.plot(positions_history[:, 0], positions_history[:, 1], 'b-', label='Worm Trajectory')
    plt.plot(stimulus_location[0], stimulus_location[1], 'r*', markersize=15, label='Stimulus Location')
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(title)
    plt.grid(True)
    plt.legend()
    plt.axis('equal')
    plt.savefig("results/figures/chemoattraction_trajectory.png")
    plt.close()
