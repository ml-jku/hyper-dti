
import os
import pickle
import random
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

import torch
from torch.utils.data import Dataset

from settings import constants
from utils.bio_encoding import precompute_protein_embeddings, precompute_molecule_embeddings


def load_chembl_data(data):
    assert False, 'Currently dataloading is disabled'


class ChEMBLData(Dataset):

    protein_embeddings = {}
    molecule_embeddings = {}

    protein_scaler = StandardScaler()
    molecule_scaler = StandardScaler()

    def __init__(self, data_path, partition='train', splitting='temporal', folds=None, debug=False, mode='pairs',
                 protein_encoder='SeqVec', molecule_encoder='CDDD', standardize=False, subset=False):
        super(ChEMBLData, self).__init__()
        self.data_path = data_path
        self.partition = partition
        self.mode = mode
        self.subset = subset
        self.folds = folds if folds is not None else {'valid': 8, 'test': 9}

        data = self.read_in()

        if mode == 'molecule':
            self.all_pids_dict = {pid: i for i, pid in enumerate(sorted(data.PID.unique()))}

        protein_embeddings = self.get_embeddings(input_type='Protein', encoder_name=protein_encoder,
                                                 unique_ids=list(data.PID.unique()),
                                                 structures=list(data['Protein'].unique()))
        molecule_embeddings = self.get_embeddings(input_type='Molecule', encoder_name=molecule_encoder,
                                                  unique_ids=list(data.MID.unique()),
                                                  structures=list(data['Molecule'].unique()))

        if splitting == 'temporal':
            data = self.temporal_splitting(data=data)
        else:
            data = self.cross_valid_splitting(strategy=splitting, data=data)

        # Unique protein IDs, held out if needed
        self.pids = list(data.PID.unique())
        self.mids = list(data.MID.unique())
        self.triplets = data[["MID", "PID", "Bioactivity"]]

        # Simplification for easier development locally
        if debug:
            self.triplets = self.triplets.head(1000)
            self.remove_empty_items()

        completeness = len(self.triplets) / (len(self.pids) * len(self.mids))
        print(f'Completeness of {partition} set of {splitting} split is: {completeness}')

        for unique_id in self.mids:
            if unique_id not in ChEMBLData.molecule_embeddings.keys():
                ChEMBLData.molecule_embeddings[unique_id] = molecule_embeddings[unique_id]
        if not standardize:
            for unique_id in self.pids:
                if unique_id not in ChEMBLData.protein_embeddings.keys():
                    ChEMBLData.protein_embeddings[unique_id] = protein_embeddings[unique_id]
        else:
            self.standardize(unique_ids=self.pids, tmp_embeddings=protein_embeddings,
                             global_embeddings=ChEMBLData.protein_embeddings, scaler=ChEMBLData.protein_scaler)
            #self.standardize(unique_ids=self.mids, tmp_embeddings=molecule_embeddings,
            #                 global_embeddings=ChEMBLData.molecule_embeddings, scaler=ChEMBLData.molecule_scaler)

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
                'protein': ChEMBLData.protein_embeddings[pid],
                'mid': molecule_batch["MID"].tolist(),
                'molecule': [ChEMBLData.molecule_embeddings[mol] for mol in molecule_batch["MID"]],
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
                'protein': [ChEMBLData.protein_embeddings[prot] for prot in molecule_batch["PID"]],
                'mid': mid,
                'molecule': ChEMBLData.molecule_embeddings[mid],
                'label': labels
            }
        else:
            batch = self.triplets.iloc[item]
            return {
                'pid': batch['PID'],
                'protein': ChEMBLData.protein_embeddings[batch['PID']],
                'mid': batch['MID'],
                'molecule': ChEMBLData.molecule_embeddings[batch["MID"]],
                'label': batch['Bioactivity']
            }

    def get_protein_memory(self, exclude_pids):
        memory = []
        for pid in self.pids:
            if pid not in exclude_pids:
                memory.append(ChEMBLData.protein_embeddings[pid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def get_molecule_memory(self, exclude_mids):
        memory = []
        mid_subset = random.choices(self.mids, k=10000)
        for mid in mid_subset:
            if mid not in exclude_mids:
                memory.append(ChEMBLData.molecule_embeddings[mid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def __len__(self):
        if self.mode == 'protein':
            return len(self.pids)
        elif self.mode == 'molecule':
            return len(self.mids)
        else:
            return len(self.triplets)

    def read_in(self):

        file_name = f"processed/data{'_small' if self.subset else ''}.pickle"

        # Process data if not already processed
        if not os.path.exists(os.path.join(self.data_path, file_name)):
            data = load_chembl_data(self.data_path)
            data.to_pickle(os.path.join(self.data_path, file_name))
        else:
            data = pd.read_pickle(os.path.join(self.data_path, file_name))

        return data.astype({"MID": "int"}, copy=False)

    def get_embeddings(self, input_type, encoder_name, unique_ids, structures):

        if not os.path.exists(os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle')):

            encoding_fn = precompute_protein_embeddings if input_type == 'Protein' else precompute_molecule_embeddings
            embeddings = encoding_fn(structures, encoder_name=encoder_name, split='full set', batch_size=16)

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
        molecule_split = {
            'train': [1977, constants.TEMPORAL_SPLIT[0]],
            'valid': [constants.TEMPORAL_SPLIT[0], constants.TEMPORAL_SPLIT[1]],
            'test': [constants.TEMPORAL_SPLIT[1], 2015]
        }

        data = data[data['Year'] >= molecule_split[self.partition][0]]
        data = data[data['Year'] < molecule_split[self.partition][1]]
        return data

    def cross_valid_splitting(self, strategy, data):

        if self.partition in self.folds.keys():
            data = data[data[strategy] == self.folds[self.partition]]
        else:  # Train on remaining folds
            train_split = np.where(
                (data[strategy] == self.folds['test']) | (data[strategy] == self.folds['valid']), False, True)
            data = data[train_split]
        return data

    def remove_empty_items(self):

        # Clean up Protein id list
        empty_ids = []
        for i in range(len(self.pids)):
            molecule_batch = self.triplets[self.triplets.PID == self.pids[i]]
            if len(molecule_batch) < 1:
                empty_ids.append(self.pids[i])
        for unique_id in empty_ids:
            self.pids.remove(unique_id)

        # Clean up Molecule id list
        empty_ids = []
        for i in range(len(self.mids)):
            molecule_batch = self.triplets[self.triplets.MID == self.mids[i]]
            if len(molecule_batch) < 1:
                empty_ids.append(self.mids[i])
        for unique_id in empty_ids:
            self.mids.remove(unique_id)

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
