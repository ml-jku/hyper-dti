
import os
import pickle
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from tdc.multi_pred import DTI

import torch
from torch.utils.data import Dataset

from settings import constants
from utils.bio_encoding import precompute_protein_embeddings, precompute_molecule_embeddings


class TdcData(Dataset):

    protein_embeddings = {}
    molecule_embeddings = {}

    protein_scaler = StandardScaler()
    molecule_scaler = StandardScaler()

    label_std = None

    def __init__(self, data_path, partition='train', splitting='temporal', folds=None, debug=False, mode='pairs',
                 protein_encoder='SeqVec', molecule_encoder='CDDD', standardize=False, label_shift=True,
                 subset=False, remove_batch=False, predefined_scaler=None):
        super().__init__()

        self.data_path = data_path
        self.dataset = data_path.split('/')[-1]
        self.partition = partition
        self.mode = mode
        self.subset = subset

        self.harmanize_mode = 'none'
        self.splitting = splitting

        data = self.read_in()

        if mode == 'molecule':
            self.all_pids_dict = {pid: i for i, pid in enumerate(sorted(data.PID.unique()))}

        protein_embeddings = self.get_embeddings(input_type='Protein', encoder_name=protein_encoder,
                                                 unique_ids=list(data.PID.unique()),
                                                 structures=list(data.Protein.unique()))
        molecule_embeddings = self.get_embeddings(input_type='Molecule', encoder_name=molecule_encoder,
                                                  unique_ids=list(data.MID.unique()),
                                                  structures=list(data.Molecule.unique()))

        data = self.read_in(partition=partition, splitting=splitting)

        # Unique protein IDs, held out if needed
        self.pids = list(data.PID.unique())
        self.mids = list(data.MID.unique())
        self.triplets = data[["MID", "PID", "Bioactivity"]]

        # Normalize labels
        if label_shift:
            if self.partition == 'train':
                TdcData.label_std = np.std(self.triplets['Bioactivity'])
            self.triplets['Bioactivity'] -= constants.BIOACTIVITY_THRESHOLD['ChEMBL']
            self.triplets['Bioactivity'] /= TdcData.label_std

        completeness = len(self.triplets) / (len(self.pids) * len(self.mids))
        print(f'Completeness of {partition} set of {splitting} split is: {completeness}')

        for unique_id in self.mids:
            if unique_id not in TdcData.molecule_embeddings.keys():
                TdcData.molecule_embeddings[unique_id] = molecule_embeddings[unique_id]
        if not standardize:
            for unique_id in self.pids:
                if unique_id not in TdcData.protein_embeddings.keys():
                    TdcData.protein_embeddings[unique_id] = protein_embeddings[unique_id]
        else:
            self.standardize(unique_ids=self.pids, tmp_embeddings=protein_embeddings,
                             global_embeddings=TdcData.protein_embeddings, scaler=TdcData.protein_scaler)
            # self.standardize(unique_ids=self.mids, tmp_embeddings=molecule_embeddings,
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
                'protein': TdcData.protein_embeddings[pid],
                'mid': molecule_batch["MID"].tolist(),
                'molecule': [TdcData.molecule_embeddings[mol] for mol in molecule_batch["MID"]],
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
                'protein': [TdcData.protein_embeddings[prot] for prot in molecule_batch["PID"]],
                'mid': mid,
                'molecule': TdcData.molecule_embeddings[mid],
                'label': labels
            }
        else:
            batch = self.triplets.iloc[item]
            return {
                'pid': batch['PID'],
                'protein': TdcData.protein_embeddings[batch['PID']],
                'mid': batch['MID'],
                'molecule': TdcData.molecule_embeddings[batch["MID"]],
                'label': batch['Bioactivity']
            }

    def get_protein_memory(self, exclude_pids):
        memory = []
        for pid in self.pids:
            if pid not in exclude_pids:
                memory.append(TdcData.protein_embeddings[pid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def get_molecule_memory(self, exclude_mids):
        memory = []
        mid_subset = random.choices(self.mids, k=10000)
        for mid in mid_subset:
            if mid not in exclude_mids:
                memory.append(TdcData.molecule_embeddings[mid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def __len__(self):
        if self.mode == 'protein':
            return len(self.pids)
        elif self.mode == 'molecule':
            return len(self.mids)
        else:
            return len(self.triplets)

    def read_in(self, partition='Full', splitting=None):

        data_cls = DTI(name=self.dataset, path=os.path.join(self.data_path, f'raw'))
        if self.harmanize_mode != 'none':
            data_cls.harmonize_affinities(self.harmanize_mode)

        if self.dataset == 'Davis':
            data_cls.convert_to_log(form='binding')

        if partition != 'Full':
            print('Temporarily splitting data randomly with built-in TDC split.')
            data_cls = data_cls.get_split()
            data = data_cls[partition]
        else:
            data = data_cls.get_data()

        data = data.rename(columns={'Drug_ID': 'MID', 'Drug': 'Molecule',
                                    'Target_ID': 'PID', 'Target': 'Protein',
                                    'Y': 'Bioactivity'})

        return data.head(10000) if self.subset else data

    def get_embeddings(self, input_type, encoder_name, unique_ids, structures):

        prepared_embedding_path = os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle')
        if not os.path.exists(prepared_embedding_path):

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

