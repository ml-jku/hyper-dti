
import time
import wandb
import argparse
import numpy as np
import pandas as pd

from baselines.tabular_baselines import TabBaselinePredictor


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Tabular Baselines in DTI Prediction')
    parser.add_argument("--data_dir", default='', type=str, help='Path to data directory.')
    parser.add_argument("--name", default='run', type=str, help='Experiment name.')
    parser.add_argument("--wandb_username", default='none', type=str)

    parser.add_argument("--split", default='lpo', type=str, choices=['random', 'temporal', 'lcco', 'lpo'],
                        help='Splitting strategy.')
    parser.add_argument("--baseline", default='RandomForest', type=str, choices=['RandomForest', 'XGBoost'])
    parser.add_argument("--objective", default='MAE', type=str, choices=['BCE', 'MSE', 'MAE'])
    parser.add_argument("--dataset", default='Lenselink', type=str, choices=['Lenselink', 'KIBA', 'Davis'])

    parser.add_argument("--molecule_encoder", default='CDDD', type=str, help='Molecular encoder.',
                        choices=['MolBert', 'CDDD'])
    parser.add_argument("--protein_encoder", default='SeqVec', type=str, help='Protein encoder.',
                        choices=['UniRep', 'SeqVec'])
    parser.add_argument("--k", default=10, type=int, choices=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10])
    args = parser.parse_args()
    config = vars(args)


    results = {}
    for fold in range(config['k']):
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

    print(f"{config['k']}-fold CV of {config['baseline']}.")
    print(f"Trained on {config['splitting']} split of {config['dataset']}.")
    print(f"With {config['molecule_encoder']} and {config['protein_encoder']} encoders.")

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

