import numpy as np
from pyneuroml import pynml
import os
import xml.etree.ElementTree as ET

class NeuralSimulator:
    def __init__(self, lems_file):
        self.lems_file = lems_file
        self.sim_time = 0
        self.lems_tree = ET.parse(lems_file)
        self.lems_root = self.lems_tree.getroot()
        self.nml_doc = self._get_neuroml_doc()


    def _get_neuroml_doc(self):
        """Finds and parses the NeuroML network file included in the LEMS file."""
        nml_file_path = None
        for include_elem in self.lems_root.findall(".//{http://www.neuroml.org/lems/0.7.0}Include"):
            if include_elem.attrib['file'].endswith('.net.nml'):
                nml_file_path = include_elem.attrib['file']
                break
        
        if not nml_file_path:
            raise ValueError("Could not find an included .net.nml file in the LEMS file.")

        base_dir = os.path.dirname(self.lems_file)
        full_nml_path = os.path.join(base_dir, nml_file_path)
        
        if not os.path.exists(full_nml_path):
            raise FileNotFoundError(f"NeuroML file not found: {full_nml_path}")

        tree = ET.parse(full_nml_path)
        return tree.getroot()


    def _get_motor_neuron_names(self):
        """
        Parses the NeuroML file to get a list of all motor neurons.
        Includes common motor neuron types like VA, DA, AS, VB, DB.
        """
        motor_neurons = []
        # Namespace is required for findall in NeuroML files
        ns = {'neuroml': 'http://www.neuroml.org/schema/neuroml2'}
        
        for population in self.nml_doc.findall('.//neuroml:population', ns):
            is_motor = False
            for prop in population.findall('.//neuroml:property', ns):
                if prop.attrib.get('tag') == 'type' and 'motor' in prop.attrib.get('value', ''):
                    is_motor = True
                    break
            if is_motor:
                neuron_id = population.attrib['id']
                # Filter for specific motor neuron classes of interest
                if any(neuron_id.startswith(prefix) for prefix in ['VA', 'DA', 'AS', 'VB', 'DB']):
                    motor_neurons.append(neuron_id)
        
        return sorted(motor_neurons)

    def advance_neural_state(self, dt_s=0.0001):
        """
        Advances the neural simulation by a single time step (dt_s, in seconds).
        """
        # Modify the LEMS file to run for a single dt
        sim_element = self.lems_root.find(".//{http://www.neuroml.org/lems/0.7.0}Simulation")
        if sim_element is None:
            # Try without namespace for simplicity if the first fails
            sim_element = self.lems_root.find(".//Simulation")

        if sim_element is not None:
            sim_element.set('length', f'{dt_s * 1000}ms') # LEMS expects time in ms or s, be explicit
            
            # Create a temporary LEMS file for the simulation in a dedicated temp folder
            temp_dir = os.path.join(os.path.dirname(self.lems_file), "temp_sim_files")
            os.makedirs(temp_dir, exist_ok=True)
            temp_lems_file = os.path.join(temp_dir, f"temp_lems_{self.sim_time}.xml")
            self.lems_tree.write(temp_lems_file)

            # Run the simulation using pyneuroml's jNeuroML interface
            # load_saved_data=False is crucial for advancing the simulation state
            pynml.run_lems_with_jneuroml(
                os.path.abspath(temp_lems_file), 
                nogui=True, 
                load_saved_data=False, 
                exec_in_dir=os.path.abspath(temp_dir)
            )

            self.sim_time += dt_s
            # Consider keeping temp files for debugging if needed
            # os.remove(temp_lems_file) 
        else:
            raise ValueError("Could not find Simulation element in LEMS file")


    def read_motor_outputs(self):
        """
        Reads the latest membrane potential of motor neurons.
        NOTE: This assumes jNeuroML outputs individual files per neuron (e.g., 'VA1_0.dat'),
        which may depend on the simulator version and LEMS file configuration.
        The primary LEMS OutputFile might consolidate all outputs into one file.
        """
        motor_neurons = self._get_motor_neuron_names()
        voltages = {}
        sim_output_dir = os.path.join(os.path.dirname(self.lems_file), "temp_sim_files")

        for neuron in motor_neurons:
            # The '0' corresponds to the instance ID of the neuron in the population
            output_file = os.path.join(sim_output_dir, f"{neuron}_0.dat")
            if os.path.exists(output_file):
                try:
                    data = np.loadtxt(output_file)
                    # Get the last voltage value
                    voltages[neuron] = data[-1, 1] if data.ndim == 2 else data[1]
                    # os.remove(output_file) # Clean up the output file
                except (IOError, IndexError) as e:
                    print(f"Warning: Could not read or parse {output_file}. Setting voltage to 0. Error: {e}")
                    voltages[neuron] = 0.0 # Resting potential as a fallback
            else:
                # If the file doesn't exist, assume a resting potential.
                voltages[neuron] = 0.0

        return np.array(list(voltages.values()))

def translate_cell_voltages(voltages):
    # Normalize voltages to a [0, 1] range for muscle activation.
    # This is a simple linear clamp and may need refinement (e.g., offset, gain, sigmoid)
    # to better reflect biological current-force relationships.
    min_v = -70  # mV (typical resting potential)
    max_v = 0    # mV (typical depolarization peak for graded neurons)
    normalized = (voltages - min_v) / (max_v - min_v)
    return np.clip(normalized, 0, 1)
