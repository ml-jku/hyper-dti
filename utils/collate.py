
import random
import numpy as np

import torch

from settings import constants


def get_collate(mode, split):
    """ Wrapper for different collate functions. """
    if mode == 'pairs':
        return collate_triplets
    elif mode == 'molecule':
        return collate_molecule
    elif split == 'train':
        return collate_protein
    else:
        return collate_protein_eval


def collate_triplets(elem_dicts):
    """ Data loading for individual drug-target-interaction triplets. """
    batch = {'pids':  torch.Tensor(), 'proteins': torch.Tensor(), 'mids': torch.Tensor(), 'molecules': torch.Tensor()}
    labels = torch.Tensor()

    for elem_dict in elem_dicts:
        label = torch.tensor(elem_dict['label']).unsqueeze(0)
        labels = label if len(labels) == 0 else torch.cat((labels, label), 0)
        mid = torch.tensor(elem_dict['mid']).unsqueeze(0)
        batch['mids'] = mid if len(batch['mids']) == 0 else torch.cat((batch['mids'], mid), 0)
        molecule = torch.tensor(elem_dict['molecule']).float().unsqueeze(0)
        batch['molecules'] = molecule if len(batch['molecules']) == 0 else torch.cat((batch['molecules'], molecule), 0)
        pid = torch.tensor(elem_dict['pid']).unsqueeze(0)
        batch['pids'] = pid if len(batch['pids']) == 0 else torch.cat((batch['pids'], pid), 0)
        protein = torch.tensor(elem_dict['protein']).float().unsqueeze(0)
        batch['proteins'] = protein if len(batch['proteins']) == 0 else torch.cat((batch['proteins'], protein), 0)

    return batch, labels


def collate_protein_single(elem_dicts):
    elem_dict = elem_dicts[0]
    labels = torch.tensor(elem_dict['label'])

    batch = {}
    batch['mids'] = torch.tensor(elem_dict['mid'])
    batch['molecules'] = torch.tensor(elem_dict['molecule']).float()
    batch['pids'] = torch.tensor(elem_dict['pid'])
    batch['proteins'] = torch.tensor(elem_dict['protein']).float()

    return batch, labels


def collate_molecule(elem_dicts):
    """ Data loading for interactions based on drug molecule. """
    batch = {'pids': torch.Tensor(), 'proteins': None, 'mids': torch.Tensor(), 'molecules': []}
    labels = torch.Tensor()

    for elem_dict in elem_dicts:
        batch_labels = torch.tensor(elem_dict['label'])
        labels = torch.cat((labels, batch_labels), 0)
        mids = torch.tensor(elem_dict['mid'])
        batch['mids'] = torch.cat((batch['mids'], mids), 0)
        molecules = torch.tensor(elem_dict['molecules']).float().unsqueeze(0)
        batch['molecules'] = molecules if len(batch['molecules']) == 0 else torch.cat((batch['molecules'], molecules), 0)
        pids = torch.tensor(elem_dict['pid'])
        batch['pids'] = torch.cat((batch['pids'], pids), 0)

    return batch, labels


def collate_protein_eval(elem_dicts):
    """ Data loading for interactions based on protein target, without over- or under-sampling. """
    return collate_protein(elem_dicts, eval=True)


def collate_protein(elem_dicts, eval=False):
    """ Data loading for interactions based on protein target. """
    batch = {'pids': torch.Tensor(), 'proteins': torch.Tensor(), 'mids': torch.Tensor(), 'molecules': []}
    labels = torch.Tensor()

    for elem_dict in elem_dicts:
        labeled_pairs = list(zip(elem_dict['mid'], elem_dict['molecule'], elem_dict['label']))
        # Oversample for low frequency or shuffle otherwise
        if not eval and len(labeled_pairs) < constants.OVERSAMPLING:
            labeled_pairs = random.choices(labeled_pairs, k=constants.OVERSAMPLING)
        elif not eval and constants.MAIN_BATCH_SIZE != -1:
            if len(labeled_pairs) < constants.MAIN_BATCH_SIZE:
                random.shuffle(labeled_pairs)
            else:
                labeled_pairs = random.sample(labeled_pairs, k=constants.MAIN_BATCH_SIZE)
        else:
            random.shuffle(labeled_pairs)
        elem_dict['mid'], elem_dict['molecule'], elem_dict['label'] = zip(*labeled_pairs)

        batch_labels = torch.tensor(elem_dict['label'])
        labels = torch.cat((labels, batch_labels), 0)

        mids = torch.tensor(elem_dict['mid'])
        batch['mids'] = torch.cat((batch['mids'], mids), 0)

        batch['molecules'].append(torch.tensor(np.stack(elem_dict['molecule'], axis=0)).float())

        pids = torch.tensor([elem_dict['pid'] for _ in range(len(elem_dict['mid']))])
        batch['pids'] = torch.cat((batch['pids'], pids), 0)

        protein = torch.tensor(elem_dict['protein']).float().unsqueeze(0)
        batch['proteins'] = protein if len(batch['proteins']) == 0 else torch.cat((batch['proteins'], protein), 0)

    return batch, labels

