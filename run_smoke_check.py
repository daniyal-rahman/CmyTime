import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'c302')))
from worm_rl_project.gym_wrapper.envs.worm_env import WormEnv
from c302.c302_Muscles import setup as c302_setup

def run_smoke_check():
    # 1. Generate NeuroML
    ref = 'Muscles'
    params = 'A'
    
    # Create a directory for the generated files
    output_dir = "smoke_check_output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    lems_file = os.path.join(output_dir, f'LEMS_c302_{params}_{ref}.xml')

    c302_setup(params,
             generate=True,
             duration=100,
             dt=0.05,
             target_directory=output_dir,
             data_reader='SpreadsheetDataReader')
    
    # Rename the generated file
    os.rename(os.path.join(output_dir, f'LEMS_c302_{params}_{ref}.xml'), lems_file)

    # Change to the output directory before running the environment
    os.chdir(output_dir)

    # 2. Launch a single-step env run
    env = WormEnv(nml_file=os.path.basename(lems_file))
    observation = env.reset()
    action = env.action_space.sample()
    new_observation, reward, _, _ = env.step(action)

    # 3. Log the motor voltages, muscle commands, and position change
    print("--- Smoke Check Results ---")
    print(f"Motor Voltages (raw): {env.simulator.read_motor_outputs()}")
    print(f"Muscle Activations (normalized): {new_observation}")
    print(f"Position Change (reward): {reward}")
    print("--------------------------")

if __name__ == '__main__':
    run_smoke_check()
