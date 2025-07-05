
import yaml
from src.connectome.loader import load_connectome
from src.models.lif_neuron import LIFNeuron
from src.models.plasticity import STDP
from src.simulation.engine import Engine
from src.simulation.runner import Runner

def main():
    with open("configs/default_hparams.yaml", 'r') as f:
        hparams = yaml.safe_load(f)

    adj_chemical, _, neuron_to_idx = load_connectome("data/raw/celegans_connectome.json")
    neurons = [LIFNeuron() for _ in range(len(neuron_to_idx))]
    stdp = STDP(**hparams)
    engine = Engine(adj_chemical, neurons, stdp)
    runner = Runner(engine, num_steps=1000)
    runner.run()

if __name__ == "__main__":
    main()
