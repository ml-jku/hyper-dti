
import sys
import time
import argparse
import numpy as np
import torch

from trainer import CrossValidator

settings = {
    'DeepPCM': {
        'seed': None,
        'dataset': 'Lenselink',
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
    },
    'HyperPCM': {
        'seed': None,
        'dataset': 'Lenselink',
        'subset': False,
        'standardize': False,
        'loss_function': 'MAE',
        'epochs': 1000,
        'patience': 100,
        'batch_size': 32,
        'oversampling': 32,
        'main_batch_size': 32,
        'learning_rate': 0.0001,
        'scheduler': 'ReduceOnPlateau',
        'lr_patience': 30,
        'lr_decay': 0.5,
        'weight_decay': 0.00001,
        'momentum': 0.8,
        'architecture': 'DeepPCM',
        'molecule_context': False,
        'protein_context': False,
        'init': 'pwi',
        'fcn_hidden_dim': 256,
        'fcn_layers': 1,
        'cls_hidden_dim': 512,
        'cls_layers': 1,
        'context_module': True,
        'hopfield_QK_dim': 512,
        'hopfield_heads': 8,
        'hopfield_beta': 0.044194173824159216,
        'hopfield_dropout': 0.5,
    }
}

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark PCM models on Lenselink dataset.')
    parser.add_argument("--model", default='HyperPCM', type=str, help='Model to benchmark.',
                        choices=['HyperPCM', 'DeepPCM'])
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
    config = settings[args.model]
    for key, value in vars(args).items():
        config[key] = value

    # Adding non-optional configurations
    config['data_dir'] = 'data'
    config['name'] = f'{args.split}/{args.name}_{time.strftime("%H%M")}'
    if config['seed'] is None or config['seed'] == -1:
        config['seed'] = np.random.randint(1, 10000)

    log = False if sys.gettrace() or config['wandb_username'] == 'none' else True    # Disable logging in Debug mode
    torch.multiprocessing.set_sharing_strategy('file_system')

    cross_validator = CrossValidator(config, log=log)
    print('Starting a 10-fold cross-validation to recreate the baseline benchmark of DeepPCM on the Lenselink dataset.')
    cross_validator.cross_validate()
