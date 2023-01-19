
import os
import json
import math
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from settings import constants
from utils.bio_encoding import precompute_protein_embeddings, precompute_molecule_embeddings


class MultiDtiData(Dataset):

    protein_embeddings = {}
    molecule_embeddings = {}

    protein_scaler = StandardScaler()
    molecule_scaler = StandardScaler()

    label_std = None

    def __init__(self, data_path, partition='train', splitting='temporal', folds=None, mode='pairs',
                 protein_encoder='SeqVec', molecule_encoder='CDDD', standardize=None,
                 label_shift=True, subset=False, remove_batch=False, predefined_scaler=None):
        super().__init__()

        self.data_path = data_path
        self.dataset = data_path.split('/')[-1]
        self.partition = partition
        self.mode = mode
        self.subset = subset
        self.valid_fold = folds['valid'] if folds is not None else 4
        self.remove_batch = remove_batch
        self.predefined_scaler = predefined_scaler
        if splitting != 'random':
            print(f'Varning: splitting strategy {splitting} is not available for Öztürk datasets, '
                  f'will default to the random split.')
        assert self.valid_fold in range(5), 'Only 5 folds available for cross-validation in Öztürk random split.'

        molecules, proteins, label_mat = self.read_in()

        if mode == 'molecule':
            self.all_pids_dict = {pid: i for i, pid in enumerate(sorted(proteins.keys()))}

        protein_embeddings = self.get_embeddings(input_type='Protein', encoder_name=protein_encoder,
                                                 unique_ids=list(proteins.keys()),
                                                 structures=list(proteins.values()))
        molecule_embeddings = self.get_embeddings(input_type='Molecule', encoder_name=molecule_encoder,
                                                  unique_ids=list(molecules.keys()),
                                                  structures=list(molecules.values()))

        data = self.get_split(list(molecules.keys()), list(proteins.keys()), label_mat, partition)

        # Unique protein IDs, held out if needed
        self.pids = list(data.PID.unique())
        self.mids = list(data.MID.unique())
        self.triplets = data[["MID", "PID", "Bioactivity"]]

        # Normalize labels
        if label_shift:
            if self.partition == 'train':
                MultiDtiData.label_std = np.std(self.triplets['Bioactivity'])
            self.triplets['Bioactivity'] -= constants.BIOACTIVITY_THRESHOLD[self.dataset]
            self.triplets['Bioactivity'] /= MultiDtiData.label_std

        completeness = len(self.triplets) / (len(self.pids) * len(self.mids))
        print(f'Completeness of {partition} set of {splitting} split is: {completeness}')

        if predefined_scaler['Molecule'] is not None:
            MultiDtiData.molecule_scaler = predefined_scaler['Molecule']
        if standardize is not None and standardize['Molecule']:
            print('Standardizing molecule embeddings.')
            self.standardize(unique_ids=self.mids, tmp_embeddings=molecule_embeddings,
                             global_embeddings=MultiDtiData.molecule_embeddings, scaler=MultiDtiData.molecule_scaler,
                             reinit=predefined_scaler['Molecule'] is None)
        else:
            for unique_id in self.mids:
                if unique_id not in MultiDtiData.molecule_embeddings.keys():
                    MultiDtiData.molecule_embeddings[unique_id] = molecule_embeddings[unique_id]

        if predefined_scaler['Protein'] is not None:
            MultiDtiData.molecule_scaler = predefined_scaler['Protein']
        if standardize is not None and standardize['Protein']:
            print('Standardizing protein embeddings.')
            self.standardize(unique_ids=self.pids, tmp_embeddings=protein_embeddings,
                             global_embeddings=MultiDtiData.protein_embeddings, scaler=MultiDtiData.protein_scaler,
                             reinit=predefined_scaler['Protein'] is None)
        else:
            for unique_id in self.pids:
                if unique_id not in MultiDtiData.protein_embeddings.keys():
                    MultiDtiData.protein_embeddings[unique_id] = protein_embeddings[unique_id]

        if constants.MAIN_BATCH_SIZE != -1 and partition == 'train':
            for i in range(len(self.pids)):
                pid = self.pids[i]
                oversample_factor = len(self.triplets[self.triplets.PID == pid]) // constants.MAIN_BATCH_SIZE
                self.pids.extend([pid for _ in range(oversample_factor)])

    def __getitem__(self, item):
        if self.mode == 'protein':
            pid = self.pids[item]
            molecule_batch = self.triplets[self.triplets.PID == pid]
            return {
                'pid': pid,
                'protein': MultiDtiData.protein_embeddings[pid],
                'mid': molecule_batch["MID"].tolist(),
                'molecule': [MultiDtiData.molecule_embeddings[mol] for mol in molecule_batch["MID"]],
                'label': molecule_batch['Bioactivity'].tolist()
            }
        elif self.mode == 'molecule':
            mid = self.mids[item]
            molecule_batch = self.triplets[self.triplets.MID == mid]
            labels = np.full([constants.NUM_TASKS], np.nan)
            for pid in molecule_batch['PID']:
                tmp_label = molecule_batch.loc[molecule_batch.PID == pid, 'Bioactivity']
                if len(tmp_label) > 1:
                    tmp_label = np.mean(tmp_label)
                labels[self.all_pids_dict[pid]] = tmp_label
            return {
                'pid': molecule_batch["PID"].tolist(),
                'protein': [MultiDtiData.protein_embeddings[prot] for prot in molecule_batch["PID"]],
                'mid': mid,
                'molecule': MultiDtiData.molecule_embeddings[mid],
                'label': labels
            }
        else:
            batch = self.triplets.iloc[item]
            return {
                'pid': batch['PID'],
                'protein': MultiDtiData.protein_embeddings[batch['PID']],
                'mid': batch['MID'],
                'molecule': MultiDtiData.molecule_embeddings[batch["MID"]],
                'label': batch['Bioactivity']
            }

    def get_protein_memory(self, exclude_pids):
        memory = []
        for pid in self.pids:
            if self.remove_batch and pid in exclude_pids:
                continue
            memory.append(MultiDtiData.protein_embeddings[pid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def get_molecule_memory(self, exclude_mids):
        memory = []
        mid_subset = random.choices(self.mids, k=10000)
        for mid in mid_subset:
            if self.remove_batch and mid in exclude_mids:
                continue
            memory.append(MultiDtiData.molecule_embeddings[mid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def __len__(self):
        if self.mode == 'protein':
            return len(self.pids)
        elif self.mode == 'molecule':
            return len(self.mids)
        else:
            return len(self.triplets)

    def read_in(self):

        molecules = json.load(open(os.path.join(self.data_path, f'raw/ligands_can.txt')))
        proteins = json.load(open(os.path.join(self.data_path, f'raw/proteins.txt')))

        label_mat = pickle.load(open(os.path.join(self.data_path, f'raw/Y'), 'rb'), encoding='latin1')
        if self.dataset == 'Davis':
            label_mat = -(np.log10(label_mat / (math.pow(10, 9))))
        return molecules, proteins, label_mat

    def get_split(self, mids, pids, label_mat, partition):
        rows, cols = np.where(np.isnan(label_mat) == False)

        if partition == 'test':
            fold = json.load(open(os.path.join(self.data_path, f'raw/test_fold_setting1.txt')))
        else:
            folds = json.load(open(os.path.join(self.data_path, f'raw/train_fold_setting1.txt')))
            if partition == 'valid':
                fold = folds[self.valid_fold]
            else:
                fold = []
                for i in range(5):
                    if i != self.valid_fold:
                        fold.extend(folds[i])

        data = pd.DataFrame({}, columns=['MID', 'PID', 'Bioactivity'])
        for ind in fold:
            data = data.append(
                {'MID': mids[rows[ind]], 'PID': pids[cols[ind]], 'Bioactivity': label_mat[rows[ind], cols[ind]]},
                ignore_index=True
            )

        return data.head(10000) if self.subset else data

    def get_embeddings(self, input_type, encoder_name, unique_ids, structures):

        prepared_embedding_path = os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle')
        if not os.path.exists(prepared_embedding_path):

            print('Warning, the following preprocessing of embeddings is not up to date. '
                  'Use precompute_embeddings.py instead.')

            encoding_fn = precompute_protein_embeddings if input_type == 'Protein' else precompute_molecule_embeddings
            embeddings = encoding_fn(structures, encoder_name=encoder_name, batch_size=16)

            embedding_dict = {}
            for item_id, emb in zip(unique_ids, embeddings):
                embedding_dict[item_id] = emb

            with open(prepared_embedding_path, 'wb') as handle:
                pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

        else:
            with open(prepared_embedding_path, 'rb') as handle:
                embeddings = pickle.load(handle)

        return embeddings

    def standardize(self, unique_ids, tmp_embeddings, global_embeddings, scaler, reinit=True):
        split_embeddings = []
        for unique_id in unique_ids:
            split_embeddings.append(tmp_embeddings[unique_id].tolist())

        if reinit and self.partition == 'train':
            assert len(split_embeddings) > 0, 'No training embeddings'
            scaler.fit(split_embeddings)
        if len(split_embeddings) > 0:
            scaled_embeddings = scaler.transform(split_embeddings)
            for unique_id, emb in zip(unique_ids, scaled_embeddings):
                if unique_id not in global_embeddings.keys():
                    global_embeddings[unique_id] = emb
        else:
            print(f'No embeddings were scaled in {self.partition} split.')

