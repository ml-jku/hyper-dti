import sys
import argparse
import torch

from settings.config import get_configs
from trainer import ChEMBLPredictor, CrossValidator


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Drug-target interaction prediction')
    config = get_configs(parser)
    log = False if sys.gettrace() or config['wandb_username'] == 'none' else True    # Disable logging in Debug mode
    torch.multiprocessing.set_sharing_strategy('file_system')

    if config['test']:
        fold = int(config['checkpoint'].split('_')[-1])
        config['folds'] = {'test': fold, 'valid': fold - 1 if fold != 0 else 9}
        trainer = ChEMBLPredictor(config, log=False)
        results = trainer.eval(checkpoint_path=config['checkpoint'])
        print({'train': results['train'], 'valid': results['valid'], 'test': results['test']})
        print(f'Optimal MCC threshold: {trainer.mcc_threshold}')
    elif config['cross_validate']:
        cross_validator = CrossValidator(config, log=log)
        if config['folds_list'] is not None:
            folds_list = [int(num) for num in config['folds_list'].split(',')]
            print(f'Starting a {len(folds_list)}-fold cross-validation, for folds {folds_list}.')
            cross_validator.cross_validate(test_folds=folds_list)
        else:
            print('Starting a 10-fold cross-validation.')
            cross_validator.cross_validate()
    else:
        print('Starting a single training and testing run.')
        trainer = ChEMBLPredictor(config, log=log)
        if log:
            with trainer.wandb_run:
                trainer.train()
                trainer.eval()
        else:
            trainer.train()
            trainer.eval()
