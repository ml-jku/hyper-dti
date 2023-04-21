
import gc
import os
import wandb
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

from hyper_dti.settings import constants
from hyper_dti.models.deep_pcm import DeepPCM
from hyper_dti.models.hyper_pcm import HyperPCM
from hyper_dti.datamodules.lenselink import ChEMBLData
from hyper_dti.datamodules.tdc import TdcData
from hyper_dti.utils.collate import get_collate
from hyper_dti.utils.setup import setup


class DtiPredictor:

    def __init__(self, config, log=True):
        super(DtiPredictor, self).__init__()

        self.log = log
        self.wandb_run = None

        if log:
            self.wandb_run = wandb.init(
                project=config['architecture'],
                group=config['name'].split('/')[0],
                name=config['name'].split('/')[1],
                config=config,
                reinit=True,
                entity=config['wandb_username']
            )

        self.config = config
        self.data_path = os.path.join(config['data_dir'], config['dataset'])
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        setup(config['seed'], config['name'])
        if config['architecture'] == 'HyperPCM':
            constants.OVERSAMPLING = config['oversampling']
            constants.MAIN_BATCH_SIZE = config['main_batch_size']
        if config['subset']:
            print('NOTE: currently running on a small subset of the data, only meant for dev and debugging!')

        standard_scaler = {'Target': None, 'Drug': None}
        batching_mode = 'target' if config['architecture'] == 'HyperPCM' else 'pairs'
        self.transfer = config['transfer']
        if self.transfer:
            data_path = os.path.join(config['data_dir'], 'Lenselink')
            pretraining_set = ChEMBLData(
                partition='train', data_path=data_path, splitting=config['split'], folds=config['folds'],
                mode=batching_mode, target_encoder=config['target_encoder'],
                drug_encoder=config['drug_encoder'], label_shift=False, subset=config['subset'],
                standardize={'Target': config['standardize_target'], 'Drug': config['standardize_drug']},
                remove_batch=config['remove_batch'], predefined_scaler=standard_scaler
            )
            standard_scaler = {'Target': pretraining_set.target_scaler, 'Drug': pretraining_set.drug_scaler}
            self.context = pretraining_set

        self.dataloaders = {}
        for split in ['train', 'valid', 'test']:
            label_shift = (self.config['loss_function'] in constants.LOSS_CLASS['regression'] and
                           not self.config['raw_reg_labels'])

            datamodule = ChEMBLData if config['dataset'] == 'Lenselink' else TdcData
            dataset = datamodule(
                partition=split, data_path=self.data_path, splitting=config['split'], folds=config['folds'],
                mode=batching_mode, target_encoder=config['target_encoder'],
                drug_encoder=config['drug_encoder'], label_shift=label_shift, subset=config['subset'],
                standardize={'Target': config['standardize_target'], 'Drug': config['standardize_drug']},
                remove_batch=config['remove_batch'], predefined_scaler=standard_scaler)
            if split == 'train' and not self.transfer:
                self.context = dataset
            collate_fn = get_collate(mode=batching_mode, split=split)
            self.dataloaders[split] = DataLoader(
                dataset, num_workers=config['num_workers'], batch_size=config['batch_size'],
                shuffle=(split == 'train'), collate_fn=collate_fn
            )

        self.model = None
        self.opt = None
        self.criterion = None
        self.scheduler = None

        # Statistics
        self.mcc_threshold = None
        self.top_valid_results = {metric: -1 for metric in constants.METRICS['classification'].keys()}
        self.top_valid_results['Loss'] = np.infty
        for metric in constants.METRICS['regression'].keys():
            self.top_valid_results[metric] = np.infty if metric in constants.LOSS_CLASS['regression'] else -1
        self.test_results = None

    def train(self):
        gc.collect()
        torch.cuda.empty_cache()

        self.init_model()
        print(self.model)
        if self.log:
            wandb.watch(self.model, log='all', log_freq=270)

        self.opt = optim.Adam(self.model.parameters(), lr=self.config['learning_rate'],
                              weight_decay=self.config['weight_decay'])
        if self.config['scheduler'] == 'ReduceLROnPlateau':
            self.scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.opt, factor=self.config['lr_decay'],
                                                                  patience=self.config['lr_patience'])

        assert self.config['loss_function'] in constants.LOSS_FUNCTION.keys(), \
            f'Error: Loss-function {self.config["loss_function"]} is not currently supported.'
        self.criterion = constants.LOSS_FUNCTION[self.config['loss_function']]()

        epoch = 0
        steps_no_improvement = 0
        for epoch in range(self.config['epochs']):
            if self.config['patience'] != -1 and steps_no_improvement > self.config['patience']:
                break

            print(f'\nEpoch {epoch}')
            results = {'train': {}, 'valid': {}}
            for split in ['train', 'valid']:
                results[split], optimal_threshold = self.run_batches(split=split)

            if self.config['scheduler'] == 'ReduceLROnPlateau':
                self.scheduler.step(results['valid']['Loss'])

            # Store result and model improvements
            for metric in results['train'].keys():
                if metric == 'Loss' and results['valid'][metric] <= self.top_valid_results[metric]:
                    if self.log:
                        wandb.run.summary[f'Best_{metric}'] = results['valid'][metric]
                    self.top_valid_results[metric] = results['valid'][metric]
                    self.mcc_threshold = optimal_threshold
                    torch.save(self.model.state_dict(), f'checkpoints/{self.config["name"]}/models/model.t7')
                    steps_no_improvement = 0
                elif metric == 'Loss':
                    steps_no_improvement += 1
                elif metric not in constants.LOSS_CLASS['regression'] and \
                    results['valid'][metric] >= self.top_valid_results[metric] or \
                        metric in constants.LOSS_CLASS['regression'] and \
                            results['valid'][metric] <= self.top_valid_results[metric]:
                    if self.log:
                        wandb.run.summary[f'Best_{metric}'] = results['valid'][metric]

                    self.top_valid_results[metric] = results['valid'][metric]
                    if metric in ['AUC', 'MCC', 'CI']:
                        torch.save(self.model.state_dict(),
                                   f'checkpoints/{self.config["name"]}/models/model_{metric}.t7')

            if self.log:
                log_dict = {}
                for metric in results['train'].keys():
                    log_dict[f'{metric}/Training'] = results['train'][metric]
                    log_dict[f'{metric}/Validation'] = results['valid'][metric]
                self.wandb_run.log(log_dict, step=epoch)

        torch.save(self.model.state_dict(), f'checkpoints/{self.config["name"]}/models/continuous/model_last.t7')
        if self.log:
            wandb.run.summary[f'Best_epoch'] = epoch - steps_no_improvement

    def eval(self, checkpoint_path=None, metric='MCC'):
        assert not (self.model is None and checkpoint_path is None), \
            'No previous run exist and no checkpoint path was given.'

        gc.collect()
        torch.cuda.empty_cache()
        name_ext = '' if metric == 'Loss' else f'_{metric}'
        results = {}

        if self.model is None:
            self.init_model()

        if checkpoint_path is not None:
            model_path = checkpoint_path
        else:
            model_path = f'checkpoints/{self.config["name"]}/models/model{name_ext}.t7'
        self.model.load_state_dict(torch.load(model_path))

        results['train'], _ = self.run_batches(split='train', opt=True, full_log=True)

        # Find optimal threshold on validation set
        results['valid'], optimal_threshold = self.run_batches(split='valid', opt=True, full_log=True)
        self.top_valid_results = results['valid']
        self.mcc_threshold = optimal_threshold

        # Test
        results['test'], _ = self.run_batches(split='test', full_log=True)
        self.test_results = results['test']

        if self.log:
            for metric, score in self.test_results.items():
                wandb.run.summary[f'Test {metric}'] = score
            wandb.run.summary[f'Optimal MCC Threshold'] = self.mcc_threshold

        return results

    def init_model(self):
        if self.config['architecture'] == 'HyperPCM':
            hyper_args = {
                'hyper_fcn': {
                    'hidden_dim': self.config['fcn_hidden_dim'],
                    'layers': self.config['fcn_layers'],
                    'selu': self.config['selu'],
                    'norm': self.config['norm'],
                    'init': self.config['init'],
                    'standardize': self.config['standardize_target']
                },
                'hopfield': {
                    'context_module': self.config['target_context'],
                    'QK_dim': self.config['hopfield_QK_dim'],
                    'heads': self.config['hopfield_heads'],
                    'beta': self.config['hopfield_beta'],
                    'dropout': self.config['hopfield_dropout'],
                    'skip': self.config['hopfield_skip']
                },
                'main_cls': {
                    'hidden_dim': self.config['cls_hidden_dim'],
                    'layers': self.config['cls_layers'],
                    'noise': not self.config['standardize_drug']
                }
            }
            memory = self.context if hyper_args['hopfield']['context_module'] else None
            self.model = HyperPCM(
                drug_encoder=self.config['drug_encoder'],
                target_encoder=self.config['target_encoder'],
                args=hyper_args,
                memory=memory
            ).to(self.device)
        else:
            assert not self.config['standardize_drug'] and not self.config['standardize_target'], \
                'Warning: Inputs should not be standardized for the original DeepPCM model.'

            args = {
                'architecture': self.config['architecture'],
                'drug_context': self.config['drug_context'],
                'target_context': self.config['target_context'],
                'hopfield': {
                    'QK_dim': self.config['hopfield_QK_dim'],
                    'heads': self.config['hopfield_heads'],
                    'beta': self.config['hopfield_beta'],
                    'dropout': self.config['hopfield_dropout'],
                    'skip': self.config['hopfield_skip'] if self.config['remove_batch'] else False
                }
            }
            memory = self.context if 'Context' in self.config['architecture'] else None
            self.model = DeepPCM(
                args=args,
                drug_encoder=self.config['drug_encoder'],
                target_encoder=self.config['target_encoder'],
                memory=memory
            ).to(self.device)

        self.model = nn.DataParallel(self.model)

    def run_batches(self, split, opt=False, full_log=False):
        self.model.train(split == 'train')

        count = 0.0
        running_loss = 0.0
        labels_true = []
        labels_prob = []
        log_dict = {'MID': [], 'PID': [], 'Label': [], 'Prediction': []}

        with torch.set_grad_enabled(split == 'train'):

            for batch, labels in tqdm(self.dataloaders[split], desc=f'    {split}: ', colour='white'):
                pids, targets, mids, drugs = batch['pids'], batch['targets'], batch['mids'], batch['drugs']
                count += labels.shape[0]
                log_dict['MID'].append(mids)
                log_dict['PID'].append(pids)
                log_dict['Label'].append(labels.cpu().numpy())
                if self.config['loss_function'] in constants.LOSS_CLASS['classification']:
                    labels = (labels > constants.BIOACTIVITY_THRESHOLD[self.config['dataset']])
                labels = labels.float().to(self.device)
                logits = self.model(batch)

                loss = self.criterion(logits, labels)
                if split == 'train':
                    loss.backward()
                    self.opt.step()
                    self.opt.zero_grad()

                running_loss += loss.item() * labels.shape[0]

                labels = labels.cpu().numpy()
                log_dict['Prediction'].append(logits.detach().cpu().numpy())
                if self.config['loss_function'] in constants.LOSS_CLASS['regression']:
                    if self.config['raw_reg_labels']:
                        labels = (labels > constants.BIOACTIVITY_THRESHOLD[self.config['dataset']]).astype(float)
                        logits -= constants.BIOACTIVITY_THRESHOLD[self.config['dataset']]
                    else:
                        labels = (labels > 0).astype(float)
                labels_true.append(labels)
                labels_prob.append(torch.sigmoid(logits).detach().cpu().numpy())

        if full_log:
            for key, values in log_dict.items():
                if 'ID' in key:
                    tmp_list = []
                    for id_batch in log_dict[key]:
                        for id_list in id_batch:
                            for id in id_list:
                                tmp_list.append(id)
                    log_dict[key] = np.array(tmp_list)
                else:
                    log_dict[key] = np.concatenate(values)
                if key == 'PID':
                    log_dict[key] = log_dict['PID'].flatten()
            log_df = pd.DataFrame(log_dict)
            log_df.to_csv(f'checkpoints/{self.config["name"]}/{split}_labels.csv', index=False)

        # Summarize batch statistics
        results = {'Loss': running_loss / count}

        mcc_threshold = None
        labels_true = np.concatenate(labels_true)
        labels_prob = np.concatenate(labels_prob)
        if self.config['loss_function'] in constants.LOSS_CLASS['regression']:
            if not full_log:
                log_dict['Label'] = np.concatenate(log_dict['Label'])
                log_dict['Prediction'] = np.concatenate(log_dict['Prediction'])
            for metric, func in constants.METRICS['regression'].items():
                if split == 'train' and metric == 'CI':
                    results[metric] = 0
                else:
                    results[metric] = func(log_dict['Label'], log_dict['Prediction'])

        for metric, func in constants.METRICS['classification'].items():
            if metric == 'MCC':
                tau = None if opt else self.mcc_threshold
                score, opt_tau = func(labels_true, labels_prob, threshold=tau)
                if split == 'valid':
                    mcc_threshold = opt_tau
                results[metric] = score
            else:
                if sum(labels_true == 1) > 0 and sum(labels_true == 0) > 0:
                    score = func(labels_true, labels_prob)
                    results[metric] = score

        return results, mcc_threshold


class CrossValidator:

    def __init__(self, config, log=True):
        super(CrossValidator, self).__init__()

        self.log = log
        self.config = config
        self.name = config['name']
        self.split = config['split']
        self.results = {'valid': {}, 'test': {}}

    def cross_validate(self, test_folds=None):
        test_folds = range(10) if test_folds is None else test_folds
        for i, fold in enumerate(test_folds):
            self.config['folds'] = {
                'test': fold,
                'valid': fold - 1 if fold != 0 else constants.MAX_FOLDS[self.config['dataset']]
            }
            self.config['name'] = f'{self.name}_test_fold_{fold}'
            self.config['seed'] = np.random.randint(1, 10000)
            tmp_result = self.parallel_training()
            self.collect_statistics(tmp_result, i)

        self.summarize()

    def collect_statistics(self, tmp_result, i):
        for split in ['valid', 'test']:
            for metric in tmp_result[split].keys():
                if i == 0:
                    self.results[split][metric] = [tmp_result[split][metric]]
                else:
                    self.results[split][metric].append(tmp_result[split][metric])

    def parallel_training(self):
        trainer = DtiPredictor(self.config, self.log)
        if self.log:
            with trainer.wandb_run:
                trainer.train()
                trainer.eval(metric=self.config['checkpoint_metric'])
        else:
            trainer.train()
            trainer.eval(metric=self.config['checkpoint_metric'])
        return {'valid': trainer.top_valid_results, 'test': trainer.test_results}

    def summarize(self):
        wandb.init(
            project=self.config['architecture'],
            group=self.config['name'].split('/')[0],
            name=f"{self.config['name'].split('/')[1]}_Summary",
            config=self.config,
            reinit=True,
            entity=self.config['wandb_username']
        )
        for stat, func in {'avg': np.mean, 'std': np.std}.items():
            for split in self.results.keys():
                for metric, values in self.results[split].items():
                    wandb.run.summary[f'{split}/{metric}/{stat}'] = func(values)

