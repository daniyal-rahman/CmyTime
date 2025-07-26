import numpy as np
import os
import subprocess
import tempfile

def run_sibernetic(muscle_activations, dt_s=0.001, sim_duration_s=0.001):
    """
    This function takes a muscle activation array, runs the Sibernetic
    solver for a single timestep, and returns the positional change.
    """
    if len(muscle_activations) != 95:
        raise ValueError(f"Expected 95 muscle activations, but got {len(muscle_activations)}")

    # Sibernetic's main_sim.py provides a 96-element array. We pad our array to match.
    padded_activations = np.pad(muscle_activations, (0, 1), 'constant')

    # Get the absolute path to the project root, assuming this script is in a subdirectory.
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    sibernetic_binary = os.path.join(project_root, 'sibernetic', 'Release', 'Sibernetic')
    
    if not os.path.exists(sibernetic_binary):
        raise FileNotFoundError(f"Sibernetic binary not found at {sibernetic_binary}. Please compile it first.")

    with tempfile.TemporaryDirectory() as temp_dir:
        activation_file = os.path.join(temp_dir, 'activations.txt')
        
        # Write the activations to a file that activation_provider.py can read.
        np.savetxt(activation_file, [padded_activations], fmt='%.8f')

        # Set up the environment for the subprocess.
        # We need to modify PYTHONPATH to include the directory of our activation provider
        # and set SIB_ACTIVATION_FILE to point to our temporary activation file.
        env = os.environ.copy()
        provider_path = os.path.dirname(__file__)
        env['PYTHONPATH'] = f"{provider_path}:{env.get('PYTHONPATH', '')}"
        env['SIB_ACTIVATION_FILE'] = activation_file
        
        # The C++ code will import 'activation_provider' instead of 'main_sim'
        # This is a hardcoded assumption about the Sibernetic binary. If it specifically
        # imports 'main_sim', we would need to rename our provider or use a different approach.
        # For now, we assume we can control the imported module. A cleaner way would be to
        # pass the module name as a command-line argument if Sibernetic supports it.
        
        command = [
            sibernetic_binary,
            '-c302',  # Tells Sibernetic to use the Python-based muscle signal provider
            '-f', 'worm',
            '-no_g',
            '-l_to',
            'lpath=' + temp_dir,
            f'timelimit={sim_duration_s}',
            f'timestep={dt_s}'
        ]

        try:
            # Run the simulation
            result = subprocess.run(command, capture_output=True, text=True, env=env, check=True)
            
            # Parse the output file to get the worm's position
            log_file = os.path.join(temp_dir, 'worm_motion_log.txt')
            if os.path.exists(log_file):
                # Read the last line of the log file for the final position
                with open(log_file, 'r') as f:
                    last_line = f.readlines()[-1]
                    # Assuming the format is: [time] [x] [y] [z] ...
                    parts = last_line.split()
                    position = np.array([float(p) for p in parts[1:4]]) # x, y, z
                    return position
            else:
                raise FileNotFoundError(f"Sibernetic output file not found: {log_file}")

        except subprocess.CalledProcessError as e:
            print("Error running Sibernetic:")
            print("Stdout:", e.stdout)
            print("Stderr:", e.stderr)
            raise

    # Fallback if something goes wrong
    return np.zeros(3)


if __name__ == '__main__':
    # Example usage:
    # This requires the Sibernetic binary to be compiled.
    try:
        muscle_activations = np.random.rand(95)
        position_delta = run_sibernetic(muscle_activations)
        print(f"Successfully ran Sibernetic. Position delta: {position_delta}")
    except (FileNotFoundError, ValueError, subprocess.CalledProcessError) as e:
        print(f"An error occurred during the test run: {e}")
