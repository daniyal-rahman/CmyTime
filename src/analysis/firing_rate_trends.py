
import matplotlib.pyplot as plt

def plot_learning_curve(rewards):
    plt.plot(rewards)
    plt.title("Learning Curve")
    plt.xlabel("Trial")
    plt.ylabel("Reward")
    plt.savefig("results/figures/learning_curve.png")
