import sys
sys.path.append('/Users/danirahman/Repos/CmyTime/worm_rl_project')
from gym_wrapper.utils import NeuralSimulator

def run_test():
    lems_file = '/Users/danirahman/Repos/CmyTime/examples/LEMS_c302_A_Muscles.xml'
    simulator = NeuralSimulator(lems_file)
    
    print("Advancing neural state...")
    simulator.advance_neural_state(dt=0.1)
    
    print("Reading motor outputs...")
    motor_outputs = simulator.read_motor_outputs()
    
    print("Motor outputs:")
    print(motor_outputs)

if __name__ == '__main__':
    run_test()
