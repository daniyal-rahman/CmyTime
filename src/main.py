from __future__ import annotations
import argparse
import scipy.sparse as sp
import numpy as np
from pathlib import Path

from connectome.exporter import load_sparse_matrix
from simulation.runner import create_default_runner


def train(adj_path: Path, steps: int = 10):
    weights = load_sparse_matrix(adj_path)
    runner = create_default_runner(weights)
    inputs = np.zeros((steps, weights.shape[0]))
    spikes = runner.run(inputs)
    return spikes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['train'])
    parser.add_argument('--adj', type=str, default='data/processed/adj_chemical.npz')
    parser.add_argument('--steps', type=int, default=10)
    args = parser.parse_args()

    if args.command == 'train':
        train(Path(args.adj), args.steps)


if __name__ == '__main__':
    main()
