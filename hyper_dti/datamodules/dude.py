
import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from hyper_dti.settings import constants


class DUDEData(Dataset):

    target_embeddings = {}
    drug_embeddings = {}

    target_scaler = StandardScaler()
    drug_scaler = StandardScaler()

    label_std = None

    def __init__(
        self, data_path, partition='train', splitting='cold-target', folds=None, mode='pairs',
        target_encoder='SeqVec', drug_encoder='CDDD', standardize=None,
        label_shift=False, subset=False, remove_batch=False, predefined_scaler=None
    ):
        super(DUDEData, self).__init__()
        self.data_path = data_path
        self.partition = partition
        self.mode = mode
        self.subset = subset
        self.folds = folds if folds is not None else {'valid': 2, 'test': 3}
        self.remove_batch = remove_batch
        
        assert mode != 'drug', 'Drug mode is not supported for the DUDE dataset.'
        assert splitting == 'cold-target', 'Only cold-target split is supported for the DUDE dataset.'
        assert label_shift == False, 'Label shift is not applicable for the DUDE dataset as it is a binary benchmark.'

        target_embeddings = self.get_embeddings(encoder_name=target_encoder)
        drug_embeddings = self.get_embeddings(encoder_name=drug_encoder)

        if partition == 'test':
            data = self.load_test()
        else: 
            data = self.load_train()    # loads either train or valid based on partition

        # Unique target IDs, held out if needed
        self.pids = list(data['Target'].unique())
        self.mids = list(data['Drug'].unique())
        self.triplets = data[['Drug', 'Target', 'Bioactivity']]

        if standardize is not None and standardize['Drug']:
            self.standardize(
                unique_ids=self.mids, tmp_embeddings=drug_embeddings, global_embeddings=DUDEData.drug_embeddings, scaler=DUDEData.drug_scaler
            )
        else:
            filtered_mids = []
            for unique_id in self.mids:
                if unique_id not in DUDEData.drug_embeddings.keys():
                    if unique_id in drug_embeddings.keys():
                        DUDEData.drug_embeddings[unique_id] = drug_embeddings[unique_id]
                        filtered_mids.append(unique_id)
            if len(filtered_mids) < len(self.mids):
                print(f'{len(self.mids)-len(filtered_mids)} drugs are removed due to missing embedding. Note, should only apply for LogpMW handcrafted encoder.')
                self.mids = filtered_mids
                self.triplets = self.triplets[self.triplets['Drug'].isin(filtered_mids)]
        if standardize is not None and standardize['Target']:
            self.standardize(
                unique_ids=self.pids, tmp_embeddings=target_embeddings, global_embeddings=DUDEData.target_embeddings, scaler=DUDEData.target_scaler
            )
        else:
            for unique_id in self.pids:
                if unique_id not in DUDEData.target_embeddings.keys():
                    DUDEData.target_embeddings[unique_id] = target_embeddings[unique_id]

        if constants.MAIN_BATCH_SIZE != -1 and partition == 'train':
            for i in range(len(self.pids)):
                pid = self.pids[i]
                oversample_factor = len(self.triplets[self.triplets['Target'] == pid]) // constants.MAIN_BATCH_SIZE
                self.pids.extend([pid for _ in range(oversample_factor)])

    def __getitem__(self, item):
        if self.mode == 'target':
            pid = self.pids[item]
            drug_batch = self.triplets[self.triplets['Target'] == pid]
            return {
                'pid': pid,
                'target': DUDEData.target_embeddings[pid],
                'mid': drug_batch['Drug'].tolist(),
                'drug': [DUDEData.drug_embeddings[mol] for mol in drug_batch['Drug']],
                'label': drug_batch['Bioactivity'].tolist()
            }
        else:
            batch = self.triplets.iloc[item]
            return {
                'pid': batch['Target'],
                'target': DUDEData.target_embeddings[batch['Target']],
                'mid': batch['Drug'],
                'drug': DUDEData.drug_embeddings[batch['Drug']],
                'label': batch['Bioactivity']
            }

    def get_target_memory(self, exclude_pids):
        memory = []
        for pid in self.pids:
            if self.remove_batch and pid in exclude_pids:  # np.array(exclude_pids)[:, 0]: Errors with different pid types
                continue
            memory.append(DUDEData.target_embeddings[pid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def get_drug_memory(self, exclude_mids):
        memory = []
        mid_subset = random.choices(self.mids, k=10000)
        for mid in mid_subset:
            if self.remove_batch and mid in exclude_mids:
                continue
            memory.append(DUDEData.drug_embeddings[mid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def __len__(self):
        if self.mode == 'target':
            return len(self.pids)
        else:
            return len(self.triplets)

    def load_train(self):
        # Load full training data
        with open(os.path.join(self.data_path, f'raw/dataPre/DUDE-foldTrain{self.folds["test"]}'), 'r') as f:
            train = f.read().strip().split('\n')
        train_dataset = [dti.strip().split() for dti in train]
        df = pd.DataFrame(train_dataset, columns=['Drug', 'Target', 'Bioactivity'])
        df = df.astype({'Bioactivity': 'int'}, copy=False)

        # Extract valid or train set from training dataset based on partition    
        targets = df['Target'].unique()
        random.seed(42)
        valid_targets = random.sample(list(targets), 10)
        df['split'] = 'train'
        df.loc[df['Target'].isin(valid_targets), 'split'] = 'valid'
        return df[df['split'] == self.partition]

    def readLinesStrip(self, lines):
        for i in range(len(lines)):
            lines[i] = lines[i].rstrip('\n')
        return lines

    def load_test(self):
        testTargetList = self.readLinesStrip(open(os.path.join(self.data_path, f'raw/dataPre/DUDE-foldTest{self.folds["test"]}')).readlines())[0].split()

        test_dataset = []
        for target in testTargetList:

            target_name = target.split('_')[0]
            targets = open(os.path.join(self.data_path, f'raw/contactMap/{target}')).readlines()
            seq = self.readLinesStrip(targets)[1]

            for drug_class in ['active', 'decoy']:
                path = os.path.join(self.data_path, f'raw/{drug_class}_smile/{target_name}_{drug_class}s_final.ism')
                drugs = open(path,'r').readlines()
                label = 1 if drug_class == 'active' else 0
                for drug in drugs:
                    test_dataset.append([drug.split(' ')[0], seq, label])
        return pd.DataFrame(test_dataset, columns=['Drug', 'Target', 'Bioactivity'])

    def get_embeddings(self, encoder_name):

        print(os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle'))
        assert os.path.exists(os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle')), f'No processed embeddings found for {encoder_name}. Use precompute_embeddings.py script.'

        with open(os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle'), 'rb') as handle:
            embeddings = pickle.load(handle)

        return embeddings

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
