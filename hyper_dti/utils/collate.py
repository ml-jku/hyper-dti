
import random
import numpy as np

import torch

from hyper_dti.settings import constants


def get_collate(mode, split):
    """ Wrapper for different collate functions. """
    if mode == 'pairs':
        return collate_triplets
    elif mode == 'drug':
        return collate_drug
    elif split == 'train':
        return collate_target
    else:
        return collate_target_eval


def collate_triplets(elem_dicts):
    """ Data loading for individual drug-target-interaction triplets. """
    batch = {'pids':  [], 'targets': torch.Tensor(), 'mids': [], 'drugs': torch.Tensor()}
    labels = torch.Tensor()

    for elem_dict in elem_dicts:
        label = torch.tensor(elem_dict['label']).unsqueeze(0)
        labels = label if len(labels) == 0 else torch.cat((labels, label), 0)
        batch['mids'].append(elem_dict['mid'])
        drug = torch.tensor(elem_dict['drug']).float().unsqueeze(0)
        batch['drugs'] = drug if len(batch['drugs']) == 0 else torch.cat((batch['drugs'], drug), 0)
        batch['pids'].append(elem_dict['pid'])
        target = torch.tensor(elem_dict['target']).float().unsqueeze(0)
        batch['targets'] = target if len(batch['targets']) == 0 else torch.cat((batch['targets'], target), 0)

    return batch, labels


def collate_target_single(elem_dicts):
    elem_dict = elem_dicts[0]
    labels = torch.tensor(elem_dict['label'])

    batch = {}
    batch['mids'] = [elem_dict['mid']]
    batch['drugs'] = torch.tensor(elem_dict['drug']).float()
    batch['pids'] = [elem_dict['pid']]
    batch['targets'] = torch.tensor(elem_dict['target']).float()

    return batch, labels


def collate_drug(elem_dicts):
    """ Data loading for interactions based on drug compound. """
    batch = {'pids': [], 'targets': None, 'mids': [], 'drugs': []}
    labels = torch.Tensor()

    for elem_dict in elem_dicts:
        batch_labels = torch.tensor(elem_dict['label'])
        labels = torch.cat((labels, batch_labels), 0)
        batch['mids'].append(elem_dict['mid'])
        drugs = torch.tensor(elem_dict['drugs']).float().unsqueeze(0)
        batch['drugs'] = drugs if len(batch['drugs']) == 0 else torch.cat((batch['drugs'], drugs), 0)
        batch['pids'].append(elem_dict['pid'])

    return batch, labels


def collate_target_eval(elem_dicts):
    """ Data loading for interactions based on protein target, without over- or under-sampling. """
    return collate_target(elem_dicts, eval=True)


def collate_target(elem_dicts, eval=False):
    """ Data loading for interactions based on protein target. """
    batch = {'pids': [], 'targets': torch.Tensor(), 'mids': [], 'drugs': []}
    labels = torch.Tensor()

    for elem_dict in elem_dicts:
        labeled_pairs = list(zip(elem_dict['mid'], elem_dict['drug'], elem_dict['label']))
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
        elem_dict['mid'], elem_dict['drug'], elem_dict['label'] = zip(*labeled_pairs)

        batch_labels = torch.tensor(elem_dict['label'])
        labels = torch.cat((labels, batch_labels), 0)

        batch['mids'].append(elem_dict['mid'])
        batch['drugs'].append(torch.tensor(np.stack(elem_dict['drug'], axis=0)).float())

        batch['pids'].append([elem_dict['pid'] for _ in range(len(elem_dict['mid']))])
        target = torch.tensor(elem_dict['target']).float().unsqueeze(0)
        batch['targets'] = target if len(batch['targets']) == 0 else torch.cat((batch['targets'], target), 0)

    return batch, labels

