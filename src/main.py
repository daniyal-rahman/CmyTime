import matplotlib.pyplot as plt
import numpy as np

from src.core.agent import Agent
from src.core.body import Body
from src.brains.static_brain import StaticBrain
from src.environments.chemoattraction_env import ChemoattractionEnv
from src.core.simulator import Simulator

def main():
    # Create the environment
    env = ChemoattractionEnv()

    # Create the brain
    brain = StaticBrain(
        connectome_path='/Users/danirahman/Repos/CmyTime/data/raw/CElegansNeuroML/NeuronConnectFormatted.xlsx',
        neuron_types_path='/Users/danirahman/Repos/CmyTime/data/raw/CElegansNeuroML/CElegansNeuronTables.xls'
    )

    # Create the body
    body = Body(initial_position=env.initial_position, initial_angle=env.initial_angle)

    # Create the agent
    agent = Agent(brain=brain, body=body)

    # Create the simulator
    simulator = Simulator(agent=agent, env=env)

    # Run the simulation
    # simulator.run(num_steps=100_000)
    simulator.run(num_steps=2_000)

    # Plot the trajectory
    positions = np.array(simulator.positions)
    plt.figure(figsize=(8, 8))
    plt.plot(positions[:, 0], positions[:, 1], label='Agent Trajectory')
    plt.scatter(env.stimulus_location[0], env.stimulus_location[1], color='red', marker='*', s=200, label='Stimulus')
    plt.scatter(env.initial_position[0], env.initial_position[1], color='green', marker='o', s=100, label='Start')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Agent Trajectory in Chemoattraction Environment')
    plt.legend()
    plt.grid(True)
    plt.savefig('results/figures/chemoattraction_trajectory.png')
    
    print(f"Initial position: {env.initial_position}")
    print(f"Final position: {positions[-1]}")
    print(f"Stimulus location: {env.stimulus_location}")
    

if __name__ == '__main__':
    main()
