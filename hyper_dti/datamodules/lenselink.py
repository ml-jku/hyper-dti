
import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from hyper_dti.settings import constants
from hyper_dti.utils.bio_encoding import precompute_target_embeddings, precompute_drug_embeddings


class ChEMBLData(Dataset):
    """
    Datasets for drug-target interactions sourced from ChEMBL.
    Currently only supporting the Lenselink version.
    """

    target_embeddings = {}
    drug_embeddings = {}

    target_scaler = StandardScaler()
    drug_scaler = StandardScaler()

    label_std = None

    def __init__(self, data_path, partition='train', splitting='temporal', folds=None, mode='pairs',
                 target_encoder='SeqVec', drug_encoder='CDDD', standardize=None,
                 label_shift=True, subset=False, remove_batch=False, predefined_scaler=None):
        super(ChEMBLData, self).__init__()
        self.data_path = data_path
        self.partition = partition
        self.mode = mode
        self.subset = subset
        self.folds = folds if folds is not None else {'valid': 8, 'test': 9}
        self.remove_batch = remove_batch

        data = self.read_in()

        if mode == 'drug':
            self.all_pids_dict = {pid: i for i, pid in enumerate(sorted(data.PID.unique()))}

        target_embeddings = self.get_embeddings(
            input_type='Target', encoder_name=target_encoder,
            unique_ids=list(data.PID.unique()), structures=list(data['Target'].unique())
        )
        drug_embeddings = self.get_embeddings(
            input_type='Drug', encoder_name=drug_encoder,
            unique_ids=list(data.MID.unique()), structures=list(data['Drug'].unique())
        )

        if splitting in ['temporal', 'leave-drug-out', 'ldo']:
            data = self.temporal_splitting(data=data)
        else:
            data = self.cross_valid_splitting(strategy=splitting, data=data)

        # Unique target IDs, drug IDs and labels
        self.pids = list(data.PID.unique())
        self.mids = list(data.MID.unique())
        self.triplets = data[["MID", "PID", "Bioactivity"]]

        # Normalize labels
        if label_shift:
            if self.partition == 'train':
                ChEMBLData.label_std = np.std(self.triplets['Bioactivity'])
            self.triplets['Bioactivity'] -= constants.BIOACTIVITY_THRESHOLD['Lenselink']
            self.triplets['Bioactivity'] /= ChEMBLData.label_std

        if standardize is not None and standardize['Drug']:
            self.standardize(unique_ids=self.mids, tmp_embeddings=drug_embeddings,
                             global_embeddings=ChEMBLData.drug_embeddings, scaler=ChEMBLData.drug_scaler)
        else:
            for unique_id in self.mids:
                if unique_id not in ChEMBLData.drug_embeddings.keys():
                    ChEMBLData.drug_embeddings[unique_id] = drug_embeddings[unique_id]
        if standardize is not None and standardize['Target']:
            self.standardize(unique_ids=self.pids, tmp_embeddings=target_embeddings,
                             global_embeddings=ChEMBLData.target_embeddings, scaler=ChEMBLData.target_scaler)
        else:
            for unique_id in self.pids:
                if unique_id not in ChEMBLData.target_embeddings.keys():
                    ChEMBLData.target_embeddings[unique_id] = target_embeddings[unique_id]

        if constants.MAIN_BATCH_SIZE != -1 and partition == 'train':
            for i in range(len(self.pids)):
                pid = self.pids[i]
                oversample_factor = len(self.triplets[self.triplets.PID == pid]) // constants.MAIN_BATCH_SIZE
                self.pids.extend([pid for _ in range(oversample_factor)])

    def __getitem__(self, item):
        if self.mode == 'target':
            pid = self.pids[item]
            drug_batch = self.triplets[self.triplets.PID == pid]
            return {
                'pid': pid,
                'target': ChEMBLData.target_embeddings[pid],
                'mid': drug_batch["MID"].tolist(),
                'drug': [ChEMBLData.drug_embeddings[mol] for mol in drug_batch["MID"]],
                'label': drug_batch['Bioactivity'].tolist()
            }
        elif self.mode == 'drug':
            mid = self.mids[item]
            drug_batch = self.triplets[self.triplets.MID == mid]
            labels = np.full([constants.NUM_TASKS], np.nan)
            for pid in drug_batch['PID']:
                tmp_label = drug_batch.loc[drug_batch.PID == pid, 'Bioactivity']
                if len(tmp_label) > 1:
                    tmp_label = np.mean(tmp_label)
                labels[self.all_pids_dict[pid]] = tmp_label
            return {
                'pid': drug_batch["PID"].tolist(),
                'target': [ChEMBLData.target_embeddings[prot] for prot in drug_batch["PID"]],
                'mid': mid,
                'drug': ChEMBLData.drug_embeddings[mid],
                'label': labels
            }
        else:
            batch = self.triplets.iloc[item]
            return {
                'pid': batch['PID'],
                'target': ChEMBLData.target_embeddings[batch['PID']],
                'mid': batch['MID'],
                'drug': ChEMBLData.drug_embeddings[batch["MID"]],
                'label': batch['Bioactivity']
            }

    def get_target_memory(self, exclude_pids):
        memory = []
        for pid in self.pids:
            if self.remove_batch and pid in exclude_pids:
                continue
            memory.append(ChEMBLData.target_embeddings[pid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def get_drug_memory(self, exclude_mids):
        memory = []
        mid_subset = random.choices(self.mids, k=10000)
        for mid in mid_subset:
            if self.remove_batch and mid in exclude_mids:
                continue
            memory.append(ChEMBLData.drug_embeddings[mid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def __len__(self):
        if self.mode == 'target':
            return len(self.pids)
        elif self.mode == 'drug':
            return len(self.mids)
        else:
            return len(self.triplets)

    def read_in(self):

        file_name = f"processed/data{'_small' if self.subset else ''}.pickle"

        assert os.path.exists(os.path.join(self.data_path, file_name)), \
            'No prepared data was found, give argument "--data_dir data" or prepare the data yourself.'

        data = pd.read_pickle(os.path.join(self.data_path, file_name))
        data = data.rename(columns={
            'MID': 'MID', 'Molecule': 'Drug', 'PID': 'PID', 'Protein': 'Target', 'Bioactivity': 'Bioactivity'
        })

        return data.astype({"MID": "int"}, copy=False)

    def get_embeddings(self, input_type, encoder_name, unique_ids, structures):

        if not os.path.exists(os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle')):

            encoding_fn = precompute_target_embeddings if input_type == 'Target' else precompute_drug_embeddings
            embeddings = encoding_fn(structures, encoder_name=encoder_name, batch_size=16)

            embedding_dict = {}
            for item_id, emb in zip(unique_ids, embeddings):
                embedding_dict[item_id] = emb

            with open(os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle'), 'wb') as handle:
                pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle'), 'rb') as handle:
                embeddings = pickle.load(handle)

        return embeddings

    def temporal_splitting(self, data):
        drug_split = {
            'train': [1977, constants.TEMPORAL_SPLIT[0]],
            'valid': [constants.TEMPORAL_SPLIT[0], constants.TEMPORAL_SPLIT[1]],
            'test': [constants.TEMPORAL_SPLIT[1], 2015]
        }

        data = data[data['Year'] >= drug_split[self.partition][0]]
        data = data[data['Year'] < drug_split[self.partition][1]]
        return data

    def cross_valid_splitting(self, strategy, data):

        if strategy in ['leave-drug-cluster-out', 'ldco']:
            strategy = 'lcco'
        elif strategy in ['leave-target-out', 'lto', 'leave-protein-out']:
            strategy = 'lpo'

        if self.partition in self.folds.keys():
            data = data[data[strategy] == self.folds[self.partition]]
        else:  # Train on remaining folds
            train_split = np.where(
                (data[strategy] == self.folds['test']) | (data[strategy] == self.folds['valid']), False, True)
            data = data[train_split]
        return data

    def standardize(self, unique_ids, tmp_embeddings, global_embeddings, scaler):
        split_embeddings = []
        for unique_id in unique_ids:
            split_embeddings.append(tmp_embeddings[unique_id].tolist())

        if self.partition == 'train':
            assert len(split_embeddings) > 0, 'No training embeddings'
            scaler.fit(split_embeddings)
        if len(split_embeddings) > 0:
            scaled_embeddings = scaler.transform(split_embeddings)
            for unique_id, emb in zip(unique_ids, scaled_embeddings):
                if unique_id not in global_embeddings.keys():
                    global_embeddings[unique_id] = emb
        else:
            print(f'No embeddings were scaled in {self.partition} split.')

