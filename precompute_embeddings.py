
import os
import pickle
import argparse
import pandas as pd
from tdc.multi_pred import DTI

from hyper_dti.utils.bio_encoding import precompute_drug_embeddings, precompute_target_embeddings


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Bio-embedding')
    parser.add_argument("--data_dir", default='', type=str, help='Path to data directory.')
    parser.add_argument("--dataset", default='Lenselink', type=str, choices=['Lenselink', 'KIBA', 'Davis'])
    parser.add_argument("--input_type", default='Drug', type=str, help='Drugs or target embeddings.',
                        choices=['Drug', 'Target'])
    parser.add_argument("--encoder_name", default='CDDD', type=str, help='Name of the encoder to use.')
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--n_jobs", default=8, type=int, choices=list(range(0, 9)))
    args = parser.parse_args()

    data_path = os.path.join(args['data_dir'], args['dataset'])

    if args.dataset == 'Lenselink':
        try:
            data = pd.read_pickle(os.path.join(data_path, 'processed/data.pickle'))
        except:
            data = pd.read_csv(os.path.join(data_path, 'processed/data.csv'))
        structures = list(data[args.input_type].unique())
        unique_ids = list(data.PID.unique()) if args.input_type == 'Target' else list(data.MID.unique())
    else:
        data_cls = DTI(name=args.dataset, path=os.path.join(data_path, f'raw'))
        data_cls.harmonize_affinities('mean')
        data = data_cls.get_data()

        data = data.rename(columns={'Drug_ID': 'MID', 'Drug': 'Drug',
                                    'Target_ID': 'PID', 'Target': 'Target',
                                    'Y': 'Bioactivity'})
        structures = list(data[args.input_type].unique())
        unique_ids = list(data.PID.unique()) if args.input_type == 'Target' else list(data.MID.unique())

    encoding_fn = precompute_target_embeddings if args.input_type == 'Target' else precompute_drug_embeddings
    embeddings = encoding_fn(structures, encoder_name=args.encoder_name, batch_size=args.batch_size, n_jobs=args.n_jobs)

    embedding_dict = {}
    for pid, emb in zip(unique_ids, embeddings):
        embedding_dict[pid] = emb

    with open(os.path.join(data_path, f'processed/{args.encoder_name}_encoding.pickle'), 'wb') as handle:
        pickle.dump(embedding_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)

