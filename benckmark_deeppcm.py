
import sys
import time
import argparse
import numpy as np
import torch

from trainer import CrossValidator

config = {
    'seed': None,
    'data_source': 'chembl',
    'dataset': 'lenselink',
    'subset': False,
    'standardize': False,
    'loss_function': 'BCE',
    'epochs': 1000,
    'patience': 100,
    'batch_size': 512,
    'learning_rate': 0.001,
    'scheduler': 'ReduceOnPlateau',
    'lr_patience': 30,
    'lr_decay': 0.5,
    'weight_decay': 0.00001,
    'momentum': 0.8,
    'architecture': 'DeepPCM',
    'molecule_context': False,
    'protein_context': False,
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark PCM models on Lenselink dataset.')

    parser.add_argument("--data_dir", default='', type=str, help='Path to data directory.')
    parser.add_argument("--name", default='benchmark_deeppcm', type=str, help='Experiment name.')
    parser.add_argument("--wandb_username", default='none', type=str)
    parser.add_argument("--split", default='lpo', type=str, help='Splitting strategy.',
                        choices=['random', 'temporal', 'leave-compound-cluster-out', 'lcco',
                                 'leave-protein-out', 'lpo'])
    parser.add_argument("--molecule_encoder", default='CDDD', type=str, help='Molecular encoder.',
                        choices=['MolBert', 'CDDD'])
    parser.add_argument("--protein_encoder", default='SeqVec', type=str, help='Protein encoder.',
                        choices=['UniRep', 'SeqVec', 'ProtTransBertBFD', 'ProtTransT5XLU50'])
    args = parser.parse_args()

    # Collected configurations
    for key, value in vars(args).items():
        config[key] = value

    # Adding non-optional configurations
    config['name'] = f'{args.split}/{args.name}_{time.strftime("%H%M")}'
    if config['seed'] is None or config['seed'] == -1:
        config['seed'] = np.random.randint(1, 10000)

    log = False if sys.gettrace() or config['wandb_username'] == 'none' else True    # Disable logging in Debug mode
    torch.multiprocessing.set_sharing_strategy('file_system')

    cross_validator = CrossValidator(config, log=log)
    print('Starting a 10-fold cross-validation to recreate the baseline benchmark of DeepPCM on the Lenselink dataset.')
    cross_validator.cross_validate()

