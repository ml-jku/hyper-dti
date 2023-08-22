
import os
import pickle
import random
import numpy as np
from sklearn.preprocessing import StandardScaler
from tdc.multi_pred import DTI

import torch
from torch.utils.data import Dataset

from hyper_dti.settings import constants
from hyper_dti.utils.bio_encoding import precompute_target_embeddings, precompute_drug_embeddings


class TdcData(Dataset):

    target_embeddings = {}
    drug_embeddings = {}

    target_scaler = StandardScaler()
    drug_scaler = StandardScaler()

    label_std = None

    def __init__(self, data_path, partition='train', splitting='cold', folds=None, mode='pairs',
                 target_encoder='SeqVec', drug_encoder='CDDD', standardize=None,
                 label_shift=True, subset=False, remove_batch=False, predefined_scaler=None):
        super().__init__()

        self.data_path = data_path
        self.dataset = data_path.split('/')[-1]
        self.partition = partition
        self.mode = mode
        self.subset = subset
        self.remove_batch = remove_batch
        self.predefined_scaler = predefined_scaler
        self.seed = folds['test'] if folds is not None else 42

        self.harmonize_mode = 'none'
        self.splitting = splitting

        # Read full dataset for quick preparation of embeddings
        full_data = self.read_in()
        if mode == 'drug':
            self.all_pids_dict = {pid: i for i, pid in enumerate(sorted(full_data.PID.unique()))}
        target_embeddings = self.get_embeddings(
            input_type='Target', encoder_name=target_encoder,
            unique_ids=list(full_data.PID.unique()), structures=list(full_data.Target.unique())
        )
        drug_embeddings = self.get_embeddings(
            input_type='Drug', encoder_name=drug_encoder,
            unique_ids=list(full_data.MID.unique()), structures=list(full_data.Drug.unique())
        )

        # Reload only the current split of the data
        data = self.read_in(partition=partition, splitting=splitting)

        # Unique target IDs, drug IDs, and labels
        self.pids = list(data.PID.unique())
        self.mids = list(data.MID.unique())
        self.triplets = data[["MID", "PID", "Bioactivity"]]

        # Normalize labels
        if label_shift:
            if self.partition == 'train':
                TdcData.label_std = np.std(self.triplets['Bioactivity'])
            self.triplets['Bioactivity'] -= constants.BIOACTIVITY_THRESHOLD[self.dataset]
            # self.triplets['Bioactivity'] /= TdcData.label_std

        if predefined_scaler['Drug'] is not None:
            TdcData.drug_scaler = predefined_scaler['Drug']
        if standardize is not None and standardize['Drug']:
            print('Standardizing drug embeddings.')
            self.standardize(
                unique_ids=self.mids, tmp_embeddings=drug_embeddings, reinit=predefined_scaler['Drug'] is None,
                global_embeddings=TdcData.drug_embeddings, scaler=TdcData.drug_scaler
            )
        else:
            for unique_id in self.mids:
                if unique_id not in TdcData.drug_embeddings.keys():
                    TdcData.drug_embeddings[unique_id] = drug_embeddings[unique_id]

        if predefined_scaler['Target'] is not None:
            TdcData.drug_scaler = predefined_scaler['Target']
        if standardize is not None and standardize['Target']:
            print('Standardizing target embeddings.')
            self.standardize(
                unique_ids=self.pids, tmp_embeddings=target_embeddings, reinit=predefined_scaler['Target'] is None,
                global_embeddings=TdcData.target_embeddings, scaler=TdcData.target_scaler
            )
        else:
            for unique_id in self.pids:
                if unique_id not in TdcData.target_embeddings.keys():
                    TdcData.target_embeddings[unique_id] = target_embeddings[unique_id]

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
                'target': TdcData.target_embeddings[pid],
                'mid': drug_batch["MID"].tolist(),
                'drug': [TdcData.drug_embeddings[mol] for mol in drug_batch["MID"]],
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
                'target': [TdcData.target_embeddings[prot] for prot in drug_batch["PID"]],
                'mid': mid,
                'drug': TdcData.drug_embeddings[mid],
                'label': labels
            }
        else:
            batch = self.triplets.iloc[item]
            return {
                'pid': batch['PID'],
                'target': TdcData.target_embeddings[batch['PID']],
                'mid': batch['MID'],
                'drug': TdcData.drug_embeddings[batch["MID"]],
                'label': batch['Bioactivity']
            }

    def get_target_memory(self, exclude_pids):
        memory = []
        for pid in self.pids:
            if self.remove_batch and pid in exclude_pids:
                continue
            memory.append(TdcData.target_embeddings[pid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def get_drug_memory(self, exclude_mids):
        memory = []
        mid_subset = random.choices(self.mids, k=10000)
        for mid in mid_subset:
            if self.remove_batch and mid in exclude_mids:
                continue
            memory.append(TdcData.drug_embeddings[mid])
        return torch.tensor(np.array(memory), dtype=torch.float32)

    def __len__(self):
        if self.mode == 'target':
            return len(self.pids)
        elif self.mode == 'drug':
            return len(self.mids)
        else:
            return len(self.triplets)

    def read_in(self, partition='Full', splitting=None):

        data_cls = DTI(name=self.dataset, path=os.path.join(self.data_path, f'raw'))
        if self.harmonize_mode != 'none':
            data_cls.harmonize_affinities(self.harmonize_mode)

        if self.dataset == 'Davis':
            data_cls.convert_to_log(form='binding')

        if partition != 'Full':
            if splitting == 'random':
                data_cls = data_cls.get_split()
            elif splitting in ['leave-drug-out', 'ldo', 'cold-drug', 'cold-molecule']:
                data_cls = data_cls.get_split(method='cold_split', column_name='Drug')
            elif splitting in ['leave-target-out', 'lto', 'lpo', 'leave-protein-out', 'cold-target', 'cold-protein']:
                data_cls = data_cls.get_split(method='cold_split', column_name='Target')
            elif splitting in ['leave-drug-target-out', 'ldto', 'cold']:
                data_cls = data_cls.get_split(method='cold_split', column_name=['Drug', 'Target'])
            else:
                assert splitting in ['random', 'cold-drug', 'cold-target', 'cold'], \
                    f'Splitting {splitting} not supported for TDC datasets, choose between [random, cold-drug, cold-target, cold]'
            data = data_cls[partition]
        else:
            data = data_cls.get_data()

        data = data.rename(columns={'Drug_ID': 'MID', 'Drug': 'Drug',
                                    'Target_ID': 'PID', 'Target': 'Target',
                                    'Y': 'Bioactivity'})

        return data.head(10000) if self.subset else data

    def get_embeddings(self, input_type, encoder_name, unique_ids, structures):

        prepared_embedding_path = os.path.join(self.data_path, f'processed/{encoder_name}_encoding.pickle')
        if not os.path.exists(prepared_embedding_path):

            encoding_fn = precompute_target_embeddings if input_type == 'Target' else precompute_drug_embeddings
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

