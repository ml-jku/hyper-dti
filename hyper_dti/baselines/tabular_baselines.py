
import os
import sys
import json
import math
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics

from hyper_dti.settings import constants

objectives = {
    'RandomForest': {
        'BCE': 'gini',
        'MSE': 'squared_error',
        'MAE': 'absolute_error',
    },
    'XGBoost': {
        'BCE': 'binary:logistic',
        'MSE': 'reg:squarederror',
        'MAE': 'reg:absoluteerror'
    }
}


class TabBaselinePredictor:

    def __init__(self, config):
        super(TabBaselinePredictor, self).__init__()

        print(f"Start {config['baseline']} predictor for fold: {config['test_fold']} ... ")

        self.dataset = config['dataset']
        self.objective = config['objective']
        self.baseline = config['baseline']
        self.seed = config['test_fold']

        self.train_x, self.train_y, self.test_x, self.test_y = self.get_data(
            data_path=os.path.join(config['data_dir'], config['dataset']),
            drug_encoder=config['drug_encoder'],
            target_encoder=config['target_encoder'],
            split=config['split'],
            test_fold=config['test_fold']
        )

        self.model = self.get_model()

        print(f"Train on {len(self.train_x)}, test on {len(self.test_x)}")

    def get_data(self, data_path, drug_encoder, target_encoder, split, test_fold):
        with open(os.path.join(data_path, f'processed/{drug_encoder}_encoding.pickle'), 'rb') as handle:
            drug_embedding = pickle.load(handle)

        with open(os.path.join(data_path, f'processed/{target_encoder}_encoding.pickle'), 'rb') as handle:
            target_embedding = pickle.load(handle)

        if self.dataset == 'Lenselink':
            data = pd.read_pickle(os.path.join(data_path, f"processed/data.pickle"))
            data = data.astype({"MID": "int"}, copy=False)

            if split == 'temporal':
                train_set = data[data['Year'] < constants.TEMPORAL_SPLIT[0]]
                test_set = data[data['Year'] >= constants.TEMPORAL_SPLIT[1]]
            else:
                train_set = data[data[split] != test_fold]
                test_set = data[data[split] == test_fold]
            train_interactions = train_set[["MID", "PID"]]
            train_y = np.array(train_set[["Bioactivity"]])
            test_interactions = test_set[["MID", "PID"]]
            test_y = np.array(test_set[["Bioactivity"]])
        else:
            dataset = self.dataset.split('/')[-1]
            from tdc.multi_pred import DTI

            data_cls = DTI(name=dataset, path=os.path.join(data_path, f'raw'))
            if dataset == 'Davis':
                data_cls.convert_to_log(form='binding')

            if split == 'random':
                data_cls = data_cls.get_split(seed=self.seed)
            elif split == 'cold_drug':
                data_cls = data_cls.get_split(method='cold_split', column_name='Drug', seed=self.seed)
            elif split == 'cold_target':
                data_cls = data_cls.get_split(method='cold_split', column_name='Target', seed=self.seed)
            elif split == 'cold':
                data_cls = data_cls.get_split(method='cold_split', column_name=['Drug', 'Target'], seed=self.seed)
            else:
                assert split in ['random', 'cold_drug', 'cold_target', 'cold'], \
                    f'Splitting {split} not supported for TDC datasets, choose between ' \
                    f'[random, cold_drug, cold_target, cold]'

            train_set = data_cls['train'].rename(
                columns={'Drug_ID': 'MID', 'Drug': 'Drug', 'Target_ID': 'PID', 'Target': 'Target',
                         'Y': 'Bioactivity'})
            train_interactions = train_set[['MID', 'PID']]
            train_y = np.array(train_set[['Bioactivity']])
            test_set = data_cls['test'].rename(
                columns={'Drug_ID': 'MID', 'Drug': 'Molecule', 'Target_ID': 'PID', 'Target': 'Protein',
                         'Y': 'Bioactivity'})
            test_interactions = test_set[['MID', 'PID']]
            test_y = np.array(test_set[['Bioactivity']])[:, 0]

        train_x = []
        for x in train_interactions.iterrows():
            train_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))
        test_x = []
        for x in test_interactions.iterrows():
            test_x.append(np.concatenate((drug_embedding[x[1]['MID']], target_embedding[x[1]['PID']])))

        return train_x, train_y, test_x, test_y

    def get_model(self):
        if self.baseline == 'RandomForest':
            from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
            model = RandomForestClassifier if self.objective in constants.LOSS_CLASS['classification'] else RandomForestRegressor
            return model(criterion=objectives[self.baseline][self.objective], verbose=1, n_jobs=-1)
        elif self.baseline == 'XGBoost':
            from xgboost import XGBClassifier, XGBRegressor
            model = XGBClassifier if self.objective in constants.LOSS_CLASS['classification'] else XGBRegressor
            return model(
                objective=objectives[self.baseline][self.objective], verbosity=1, n_jobs=-1, seed=self.seed,
                #eta=0.01, max_depth=15, tree_method='gpu_hist'
                colsample_bytree=0.8, learning_rate=0.03, max_depth=19, alpha=5, n_estimators=855, min_child_weight=5,
            )
        else:
            print(f'Baseline {self.baseline} not supported.')
            sys.exit()

    def run(self):
        score = {}
        if self.objective in constants.LOSS_CLASS['classification']:
            self.train_y = (self.train_y > constants.BIOACTIVITY_THRESHOLD[self.dataset])
            self.test_y = (self.test_y > constants.BIOACTIVITY_THRESHOLD[self.dataset])
        else:
            self.train_y -= constants.BIOACTIVITY_THRESHOLD[self.dataset]
            self.test_y -= constants.BIOACTIVITY_THRESHOLD[self.dataset]
        self.model.fit(self.train_x, self.train_y)
        train_pred = self.model.predict(self.train_x)
        test_pred = self.model.predict(self.test_x)
        if self.objective in constants.LOSS_CLASS['regression']:
            for metric, score_func in metrics['regression'].items():
                #train_score = score_func(self.train_y, train_pred)
                test_score = score_func(self.test_y, test_pred)
                score[metric] = {'train': -100, 'test': test_score}
            self.train_y = (self.train_y > 0)
            self.test_y = (self.test_y > 0)
            train_pred = sigmoid(train_pred.astype(float))
            test_pred = sigmoid(test_pred.astype(float))

        for metric, score_func in metrics['classification'].items():
            if metric == 'MCC':
                train_score, mcc_threshold = score_func(self.train_y, train_pred, threshold=None)
                test_score, _ = score_func(self.test_y, test_pred, threshold=mcc_threshold)
            else:
                train_score = score_func(self.train_y, train_pred)
                test_score = score_func(self.test_y, test_pred)
            score[metric] = {'train': train_score, 'test': test_score}
        return score


def sigmoid(x):
    return 1 / (1 + np.exp(-x))

