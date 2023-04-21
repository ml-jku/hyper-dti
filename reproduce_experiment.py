
import argparse
from hyper_dti.experiment import reproduce_hyperpcm, reproduce_deeppcm, reproduce_tabular


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Benchmark PCM models on Lenselink dataset.')
    parser.add_argument("--model", default='HyperPCM', type=str, help='Model to benchmark.',
                        choices=['HyperPCM', 'DeepPCM', 'RandomForest', 'XGBoost'])
    parser.add_argument("--name", default='benchmark_hyperpcm', type=str, help='Experiment name.')
    parser.add_argument("--wandb_username", default='none', type=str)
    parser.add_argument("--dataset", default='Lenselink', type=str, choices=['Lenselink', 'Davis'])
    parser.add_argument("--split", default='lpo', type=str, help='Splitting strategy.',
                        choices=['random', 'temporal', 'leave-compound-cluster-out', 'lcco',
                                 'leave-protein-out', 'lpo', 'cold-drug', 'cold-target', 'cold'])
    parser.add_argument("--drug_encoder", default='CDDD', type=str, help='Drug encoder.',
                        choices=['MolBert', 'CDDD'])
    parser.add_argument("--target_encoder", default='SeqVec', type=str, help='Target encoder.',
                        choices=['UniRep', 'SeqVec', 'ProtBert', 'ProtT5'])
    args = parser.parse_args()

    if args['model'] == 'HyperPCM':
        reproduce_hyperpcm(
            dataset=args['dataset'], split=args['split'], drug_encoder=args['drug_encoder'],
            target_encoder=args['target_encoder'], name=args['name'], wandb_username=args['wandb_username']
        )
    elif args['model'] == 'DeepPCM':
        reproduce_deeppcm(
            dataset=args['dataset'], split=args['split'], drug_encoder=args['drug_encoder'],
            target_encoder=args['target_encoder'], name=args['name'], wandb_username=args['wandb_username']
        )
    else:
        reproduce_tabular(
            baseline=args['model'], dataset=args['dataset'], split=args['split'], drug_encoder=args['drug_encoder'],
            target_encoder=args['target_encoder'], name=args['name'], wandb_username=args['wandb_username']
        )

