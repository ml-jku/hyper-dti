
import os
import sys
import json
import math
import pickle
import numpy as np
import pandas as pd
from sklearn import metrics

from settings import constants

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
            molecule_encoder=config['molecule_encoder'],
            protein_encoder=config['protein_encoder'],
            split=config['split'],
            test_fold=config['test_fold']
        )

        self.model = self.get_model()

        print(f"Train on {len(self.train_x)}, test on {len(self.test_x)}")

    def get_data(self, data_path, molecule_encoder, protein_encoder, split, test_fold):
        with open(os.path.join(data_path, f'processed/{molecule_encoder}_encoding.pickle'), 'rb') as handle:
            molecule_embedding = pickle.load(handle)

        with open(os.path.join(data_path, f'processed/{protein_encoder}_encoding.pickle'), 'rb') as handle:
            protein_embedding = pickle.load(handle)

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
            train_y = train_set[["Bioactivity"]]
            test_interactions = test_set[["MID", "PID"]]
            test_y = test_set[["Bioactivity"]]
        elif 'tdc' in self.dataset:
            dataset = self.dataset.split('/')[1]
            from tdc.multi_pred import DTI

            data = DTI(name=dataset, path=os.path.join(data_path, f'raw'))
            if dataset == 'Davis':
                data.convert_to_log(form='binding')
            data = data.get_split()
            train_set = data['train'].rename(
                columns={'Drug_ID': 'MID', 'Drug': 'Molecule', 'Target_ID': 'PID', 'Target': 'Protein',
                         'Y': 'Bioactivity'})
            train_interactions = train_set[['MID', 'PID']]
            train_y = train_set[['Bioactivity']]
            test_set = data['test'].rename(
                columns={'Drug_ID': 'MID', 'Drug': 'Molecule', 'Target_ID': 'PID', 'Target': 'Protein',
                         'Y': 'Bioactivity'})
            test_interactions = test_set[['MID', 'PID']]
            test_y = test_set[['Bioactivity']]
        else:
            molecules = json.load(open(os.path.join(data_path, "raw/ligands_can.txt")))
            proteins = json.load(open(os.path.join(data_path, "raw/proteins.txt")))
            mids = list(molecules.keys())
            pids = list(proteins.keys())

            label_mat = pickle.load(open(os.path.join(data_path, f'raw/Y'), 'rb'), encoding='latin1')
            if self.dataset == 'Davis':
                label_mat = -(np.log10(label_mat / (math.pow(10, 9))))

            rows, cols = np.where(np.isnan(label_mat) == False)

            folds = json.load(open(os.path.join(data_path, f'raw/train_fold_setting1.txt')))
            train_fold = []
            for i in range(5):
                if i != test_fold:
                    train_fold.extend(folds[i])
            train_interactions = pd.DataFrame({}, columns=['MID', 'PID'])
            train_y = []
            for ind in train_fold:
                train_interactions = pd.concat(
                    [train_interactions, pd.DataFrame({'MID': mids[rows[ind]], 'PID': pids[cols[ind]]}, index=[0])],
                    ignore_index=True
                )
                train_y.append(label_mat[rows[ind], cols[ind]])

            test_fold = json.load(open(os.path.join(data_path, f'raw/test_fold_setting1.txt')))
            test_interactions = pd.DataFrame({}, columns=['MID', 'PID'])
            test_y = []
            for ind in test_fold:
                test_interactions = pd.concat(
                    [test_interactions, pd.DataFrame({'MID': mids[rows[ind]], 'PID': pids[cols[ind]]}, index=[0])],
                    ignore_index=True
                )
                test_y.append(label_mat[rows[ind], cols[ind]])

        train_x = []
        for x in train_interactions.iterrows():
            train_x.append(np.concatenate((molecule_embedding[x[1]['MID']], protein_embedding[x[1]['PID']])))
        test_x = []
        for x in test_interactions.iterrows():
            test_x.append(np.concatenate((molecule_embedding[x[1]['MID']], protein_embedding[x[1]['PID']])))

        return train_x, np.array(train_y), test_x, np.array(test_y)

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
        self.model.fit(self.train_x, self.train_y)
        train_pred = self.model.predict(self.train_x)
        test_pred = self.model.predict(self.test_x)
        if self.objective in constants.LOSS_CLASS['regression']:
            for metric, score_func in metrics['regression'].items():
                #train_score = score_func(self.train_y, train_pred)
                test_score = score_func(self.test_y, test_pred)
                score[metric] = {'train': -100, 'test': test_score}
            self.train_y = (self.train_y > constants.BIOACTIVITY_THRESHOLD[self.dataset])
            self.test_y = (self.test_y > constants.BIOACTIVITY_THRESHOLD[self.dataset])
            train_pred -= constants.BIOACTIVITY_THRESHOLD[self.dataset]
            test_pred -= constants.BIOACTIVITY_THRESHOLD[self.dataset]
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

