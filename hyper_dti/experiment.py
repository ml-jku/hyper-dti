
import sys
import time
import wandb
import numpy as np
import pandas as pd
import torch

from hyper_dti.trainer import CrossValidator
from hyper_dti.baselines.tabular_baselines import TabBaselinePredictor


def reproduce_hyperpcm(dataset='Lenselink', split='lpo', drug_encoder='CDDD', target_encoder='SeqVec',
                       name='exp', wandb_username='none'):
    """
    Function to reproduce experiments with HyperPCM model on given dataset and data split.
    """
    config = {
        'seed': np.random.randint(1, 10000),
        'name': f'{split}/{name}_{time.strftime("%H%M")}',
        'wandb_username': wandb_username,
        'data_dir': f'hyper_dti/data',
        'dataset': dataset,
        'subset': False,
        'transfer': False,
        'split': split,
        'drug_encoder': drug_encoder,
        'target_encoder': target_encoder,
        'standardize_drug': False,
        'standardize_target': True,
        'loss_function': 'MAE' if dataset == 'Leneslink' else 'MSE',
        'raw_reg_labels': dataset != 'Davis',
        'checkpoint_metric': 'MCC' if dataset == 'Lenselink' else 'MSE',
        'epochs': 1000,
        'patience': 100,
        'batch_size': 32,
        'oversampling': 32,
        'main_batch_size': 32,
        'learning_rate': 0.0000005 if dataset == 'DUDE' else 0.0001,
        'scheduler': 'ReduceOnPlateau',
        'lr_patience': 30,
        'lr_decay': 0.5,
        'weight_decay': 0.00005 if dataset == 'DUDE' else 0.00001,
        'momentum': 0.8,
        'num_workers': 0,           # Should be 4 but currently not working
        'architecture': 'HyperPCM',
        'drug_context': False,
        'target_context': True,
        'init': 'pwi',
        'norm': None,
        'selu': False,
        'fcn_hidden_dim': 256,
        'fcn_layers': 1,
        'cls_hidden_dim': 1024,  # Corresponds to 512 hidden units due to an updated FCN implementation
        'cls_layers': 1 if dataset == 'Lenselink' else 2,
        'hopfield_QK_dim': 512,
        'hopfield_heads': 8,
        'hopfield_beta': 0.044194173824159216,  # Sqrt(1/QK_dim)
        'hopfield_dropout': 0.5,
        'hopfield_skip': dataset == 'Lenselink',
        'remove_batch': dataset == 'Lenselink'
    }

    log = False if sys.gettrace() or config['wandb_username'] == 'none' else True  # Disable logging in Debug mode
    torch.multiprocessing.set_sharing_strategy('file_system')

    cross_validator = CrossValidator(config, log=log)
    if dataset == 'Lenselink':
        k = 10
    elif dataset == 'Davis':
        k = 5
    else:
        k = 3
    print(f'Recreate {k}-fold cross-validation of HyperPCM on the {split} split of the {dataset} benchmark.')
    cross_validator.cross_validate()


def reproduce_deeppcm(dataset='Leselink', split='lpo', drug_encoder='CDDD', target_encoder='SeqVec',
                      name='exp', wandb_username='none'):
    """
    Function to reproduce experiments with DeepPCM model on given dataset and data split.
    """
    config = {
        'seed': np.random.randint(1, 10000),
        'name': f'{split}/{name}_{time.strftime("%H%M")}',
        'wandb_username': wandb_username,
        'data_dir': f'hyper_dti/data',
        'dataset': dataset,
        'subset': False,
        'transfer': False,
        'split': split,
        'drug_encoder': drug_encoder,
        'target_encoder': target_encoder,
        'standardize_drug': False,
        'standardize_target': False,
        'loss_function': 'BCE' if dataset == 'Leneslink' else 'MSE',
        'raw_reg_labels': dataset == 'Lenselink',
        'checkpoint_metric': 'MCC' if dataset == 'Lenselink' else 'MSE',
        'epochs': 1000,
        'patience': 100,
        'batch_size': 512,
        #'oversampling': 32,
        #'main_batch_size': 32,
        'learning_rate': 0.001,
        'scheduler': 'ReduceOnPlateau',
        'lr_patience': 30,
        'lr_decay': 0.5,
        'weight_decay': 0.00001,
        'momentum': 0.8,
        'num_workers': 4,           # Should be 4 but currently not working
        'architecture': 'DeepPCM',
        'drug_context': False,
        'target_context': False,
        #'init': 'pwi',
        #'norm': None,
        #'selu': False,
        #'fcn_hidden_dim': 256,
        #'fcn_layers': 1,
        #'cls_hidden_dim': 1024,  # Corresponds to 512 hidden units due to an updated FCN implementation
        #'cls_layers': 1 if dataset == 'Lenselink' else 2,
        'hopfield_QK_dim': 512,
        'hopfield_heads': 8,
        'hopfield_beta': 0.044194173824159216,  # Sqrt(1/QK_dim)
        'hopfield_dropout': 0.5,
        'hopfield_skip': dataset == 'Lenselink',
        'remove_batch': dataset == 'Lenselink'
    }
    
    assert dataset == 'Lenselink', \
        'DeepPCM only applies to the Lenselink benchmarks in the manuscript.'

    log = False if sys.gettrace() or config['wandb_username'] == 'none' else True  # Disable logging in Debug mode
    torch.multiprocessing.set_sharing_strategy('file_system')

    cross_validator = CrossValidator(config, log=log)
    k = 10 if dataset == 'Lenselink' else 5
    print(f'Recreate {k}-fold cross-validation of DeepPCM on the {split} split of the {dataset} benchmark.')
    cross_validator.cross_validate()


def reproduce_tabular(baseline='XGBoost', dataset='Leselink', split='lpo', drug_encoder='CDDD', target_encoder='SeqVec',
                     name='exp', wandb_username='none'):
    """
    Function to reproduce experiments with given tabular baseline on given dataset and data split.
    """
    config = {
        'name': f'{split}/{name}_{time.strftime("%H%M")}',
        'wandb_username': wandb_username,
        'data_dir': f'hyper_dti/data',
        'dataset': dataset,
        'split': split,
        'drug_encoder': drug_encoder,
        'target_encoder': target_encoder,
        'objective': 'BCE' if dataset == 'Leneslink' else 'MSE',
        'baseline': baseline
    }
    
    assert dataset in ['Lenselink', 'Davis'], \
        'Baselines were only run on the Lenselink and Davis benchmarks in the manuscript.'

    k = 10 if dataset == 'Lenselink' else 5
    results = {}
    for fold in range(k):
        config['test_fold'] = fold
        score = TabBaselinePredictor(config).run()
        for metric, scores in score.items():
            if metric not in results.keys():
                results[metric] = {'train': [], 'test': []}
            results[metric]['train'].append(scores['train'])
            results[metric]['test'].append(scores['test'])

    wandb.init(
        project='TabularBaselines',
        group=config['baseline'],
        name=f"{config['name']}_{time.strftime('%H%M')}",
        config=config,
        reinit=True,
        entity=config['wandb_username']
    )

    print(f"{k}-fold CV of {config['baseline']}.")
    print(f"Trained on {config['splitting']} split of {config['dataset']}.")
    print(f"With {config['drug_encoder']} and {config['target_encoder']} encoders.")

    score_df = {metric: [] for metric in results.keys()}
    for metric, scores in results.items():
        for split in ['train', 'test']:
            mean = np.mean(scores[split])
            std = np.std(scores[split])
            wandb.run.summary[f'{metric}_{split}/avg'] = mean
            wandb.run.summary[f'{metric}_{split}/std'] = std
            score_df[metric].append(f"{mean:.3f}±{std:.3f}")
    score_df = pd.DataFrame(score_df, index=['train', 'test'])
    print(score_df.to_string())

