import matplotlib.pyplot as plt
import matplotlib.cm as cm # Import colormap
import numpy as np
import argparse
import imageio
import os

from src.core.agent import Agent
from src.core.body import Body
from src.brains.static_brain import StaticBrain
from src.brains.learning_brain import LearningBrain
from src.environments.chemoattraction_env import ChemoattractionEnv
from src.environments.trapped_chemoattraction_env import TrappedChemoattractionEnv
from src.core.simulator import Simulator

def main():
    parser = argparse.ArgumentParser(description="Run a C. elegans simulation.")
    parser.add_argument("--brain", type=str, default="static", help="Brain to use (static or learning)")
    parser.add_argument("--env", type=str, default="chemoattraction", help="Environment to use (chemoattraction or trapped)")
    parser.add_argument("--epochs", type=int, default=1, help="Number of training epochs")
    parser.add_argument("--steps", type=int, default=2000, help="Number of steps per epoch")
    args = parser.parse_args()

    # Create the environment
    if args.env == "trapped":
        env = TrappedChemoattractionEnv()
    else:
        env = ChemoattractionEnv()

    # Create the brain
    if args.brain == "learning":
        brain = LearningBrain(
            connectome_path='/Users/danirahman/Repos/CmyTime/data/raw/CElegansNeuroML/NeuronConnectFormatted.xlsx',
            neuron_types_path='/Users/danirahman/Repos/CmyTime/data/raw/CElegansNeuroML/CElegansNeuronTables.xls'
        )
    else:
        brain = StaticBrain(
            connectome_path='/Users/danirahman/Repos/CmyTime/data/raw/CElegansNeuroML/NeuronConnectFormatted.xlsx',
            neuron_types_path='/Users/danirahman/Repos/CmyTime/data/raw/CElegansNeuroML/CElegansNeuronTables.xls'
        )

    # Create the body and agent
    body = Body(initial_position=env.initial_position, initial_angle=env.initial_angle)
    agent = Agent(brain=brain, body=body)

    # Create the simulator
    simulator = Simulator(agent=agent, env=env)

    # --- Simulation and Plotting ---
    filenames = []
    all_trajectories = []
    
    for epoch in range(args.epochs):
        print(f"--- Epoch {epoch+1}/{args.epochs} ---")
        simulator.run(num_steps=args.steps)
        all_trajectories.append(np.array(simulator.positions))
        print(f"Final position for epoch {epoch+1}: {simulator.positions[-1]}")
        print(f"Time in trap for epoch {epoch+1}: {simulator.time_in_trap_history[-1]} steps")
        print(f"Total trap penalty for epoch {epoch+1}: {simulator.total_trap_penalty_history[-1]:.2f}")
        
        # Create plot for the current epoch
        fig, ax = plt.subplots(figsize=(10, 10))
        fig.patch.set_facecolor('#212121')
        ax.set_facecolor('#212121')

        # Plot previous trajectories as faint lines
        for i in range(epoch):
            ax.plot(all_trajectories[i][:, 0], all_trajectories[i][:, 1], color='gray', alpha=0.15, linewidth=1)

        # Plot current trajectory
        current_positions = all_trajectories[epoch]
        ax.plot(current_positions[:, 0], current_positions[:, 1], color='#64FFDA', linewidth=2, label=f'Epoch {epoch+1}')

        # Plot environment objects
        ax.scatter(env.stimulus_location[0], env.stimulus_location[1], color='#FF00FF', marker='*', s=400, zorder=5, label='Stimulus')
        if hasattr(env, 'traps'):
            for trap in env.traps:
                trap_circle = plt.Circle(trap['location'], trap['radius'], color='#FF4500', alpha=0.5, zorder=4, edgecolor='black', linewidth=1.5)
                ax.add_artist(trap_circle)
        ax.scatter(env.initial_position[0], env.initial_position[1], color='#00FF00', marker='o', s=150, zorder=5, label='Start')
        
        # Style the plot
        ax.set_xlabel('X Position', color='white')
        ax.set_ylabel('Y Position', color='white')
        ax.set_title(f'Agent Trajectory (Epoch {epoch+1}/{args.epochs})', color='white', fontsize=16)
        ax.tick_params(axis='x', colors='white')
        ax.tick_params(axis='y', colors='white')
        ax.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax.set_xlim([-20, 120]) # Set consistent X limits
        ax.set_ylim([-20, 120]) # Set consistent Y limits
        ax.set_aspect('equal', adjustable='box') # Ensure aspect ratio is equal
        ax.legend(loc='upper left', facecolor='#212121', edgecolor='white', labelcolor='white') # Adjust legend
        
        # Save frame
        filename = f"results/figures/frame_{epoch+1}.png"
        filenames.append(filename)
        plt.savefig(filename, facecolor=fig.get_facecolor())
        plt.close(fig)
        
        simulator.env.reset()

    # --- Create GIF ---
    gif_path = f'results/figures/{args.brain}_{args.env}_trajectory.gif'
    with imageio.get_writer(gif_path, mode='I', duration=0.5) as writer:
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)
    
    # --- Cleanup ---
    for filename in filenames:
        os.remove(filename)

    print(f"\n--- GIF created at {gif_path} ---")

    # --- Create Companion Plot with Gradient ---
    if args.epochs > 1:
        fig_comp, ax_comp = plt.subplots(figsize=(10, 10))
        fig_comp.patch.set_facecolor('#212121')
        ax_comp.set_facecolor('#212121')

        cmap = cm.viridis # Choose a colormap
        for i, trajectory in enumerate(all_trajectories):
            color = cmap(i / (len(all_trajectories) - 1)) if len(all_trajectories) > 1 else cmap(0.5)
            ax_comp.plot(trajectory[:, 0], trajectory[:, 1], color=color, linewidth=1.5, alpha=0.7, label=f'Epoch {i+1}')

        # Plot environment objects on the companion plot
        ax_comp.scatter(env.stimulus_location[0], env.stimulus_location[1], color='#FF00FF', marker='*', s=400, zorder=5, label='Stimulus')
        if hasattr(env, 'traps'):
            for trap in env.traps:
                trap_circle = plt.Circle(trap['location'], trap['radius'], color='#FF4500', alpha=0.5, zorder=4, edgecolor='black', linewidth=1.5)
                ax_comp.add_artist(trap_circle)
        ax_comp.scatter(env.initial_position[0], env.initial_position[1], color='#00FF00', marker='o', s=150, zorder=5, label='Start')

        # Style the companion plot
        ax_comp.set_xlabel('X Position', color='white')
        ax_comp.set_ylabel('Y Position', color='white')
        ax_comp.set_title('Agent Trajectories Over Epochs (Gradient)', color='white', fontsize=16)
        ax_comp.tick_params(axis='x', colors='white')
        ax_comp.tick_params(axis='y', colors='white')
        ax_comp.grid(True, color='gray', linestyle='--', linewidth=0.5, alpha=0.3)
        ax_comp.set_xlim([-20, 120])
        ax_comp.set_ylim([-20, 120])
        ax_comp.set_aspect('equal', adjustable='box')
        
        # Add a colorbar
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=plt.Normalize(vmin=0, vmax=args.epochs - 1))
        sm.set_array([])
        cbar = fig_comp.colorbar(sm, ax=ax_comp, orientation='vertical', fraction=0.02, pad=0.04)
        cbar.set_label('Epoch', color='white')
        cbar.ax.yaxis.set_tick_params(color='white')
        plt.setp(cbar.ax.get_yticklabels(), color='white')

        companion_plot_path = f'results/figures/{args.brain}_{args.env}_all_epochs_trajectory_gradient.png'
        plt.savefig(companion_plot_path, facecolor=fig_comp.get_facecolor())
        plt.close(fig_comp)
        print(f"\n--- Companion plot created at {companion_plot_path} ---")

if __name__ == '__main__':
    main()