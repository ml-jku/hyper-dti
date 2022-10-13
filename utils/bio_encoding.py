import gc
import sys
import numpy as np
from tqdm import tqdm
from typing import Any, Dict, Union, Callable
from concurrent.futures import ThreadPoolExecutor


def encode_molecule(batch, encoder, name):
    """ Wraps encoding of drug compounds, i.e. molecules, from different encoders. """
    embeddings = encoder(batch)
    return embeddings if name == 'CDDD' else embeddings[0]


def encode_protein(batch, encoder):
    """ Wraps encoding of targets, i.e. proteins, from different encoders. """
    full_embeddings = encoder.embed_batch(batch)
    embeddings = np.array([])
    for emb in full_embeddings:
        emb = encoder.reduce_per_protein(emb)
        emb = np.expand_dims(emb, axis=0)
        embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)
    return embeddings


# Use re-implementation of UniRep
class UniRepEmbedder:

    _params: Dict[str, Any]

    def __init__(self):
        from jax_unirep.utils import load_params
        self._params = load_params()

    def embed_batch(self, batch):
        from jax_unirep.featurize import get_reps
        h, h_final, c_final = get_reps(batch)
        return h

    @staticmethod
    def reduce_per_protein(embedding):
        return embedding.mean(axis=0)


def precompute_molecule_embeddings(molecules, encoder_name, split, batch_size):
    gc.collect()

    if encoder_name == 'CDDD':          # Server conda env cddd not work on server
        from cddd.inference import InferenceModel
        CDDD_MODEL_DIR = 'checkpoints/CDDD/default_model'
        cddd_model = InferenceModel(CDDD_MODEL_DIR)
        mol_encoder = cddd_model.seq_to_emb
    elif encoder_name == 'MolBert':     # Server conda env molbert
        from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
        MOLBERT_MODEL_DIR = 'checkpoints/MolBert/molbert_100epochs/checkpoints/last.ckpt'
        molbert_model = MolBertFeaturizer(MOLBERT_MODEL_DIR, max_seq_len=500, embedding_type='average-1-cat-pooled')
        mol_encoder = molbert_model.transform
    elif encoder_name == 'ECFP':         # Server conda env molbert
        from molbert.utils.featurizer.molfeaturizer import MorganFPFeaturizer
        ecfp_model = MorganFPFeaturizer(fp_size=2048, radius=2, use_counts=True, use_features=False)
        mol_encoder = ecfp_model.transform
    elif encoder_name == 'RDKit':        # Server conda env molbert? ERROR
        from molbert.utils.featurizer.molfeaturizer import PhysChemFeaturizer
        rdkit_norm_model = PhysChemFeaturizer(normalise=True)
        mol_encoder = rdkit_norm_model.transform
    else:
        print(f'Protein encoder {encoder_name} currently not supported.')
        sys.exit()

    embeddings = np.array([])
    with ThreadPoolExecutor(max_workers=8) as executor:
        desc = f'Pre-computing molecule encodings with {encoder_name} for {split}: '
        batches = (molecules[i:i + batch_size] for i in range(0, len(molecules), batch_size))
        threads = [executor.submit(encode_molecule, batch, mol_encoder, encoder_name) for batch in batches]
        for t in tqdm(threads, desc=desc, colour='white'):
            emb = t.result()
            embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)
    return embeddings


def precompute_protein_embeddings(proteins, encoder_name, split, batch_size):
    gc.collect()

    if encoder_name == 'ProtTransBertBFD':      # Server Conda env bio_embeddings
        from bio_embeddings.embed import ProtTransBertBFDEmbedder
        bio_encoder = ProtTransBertBFDEmbedder()
    elif encoder_name == 'ProtTransT5XLU50':    # Server Conda env bio_embeddings
        from bio_embeddings.embed import ProtTransT5XLU50Embedder
        bio_encoder = ProtTransT5XLU50Embedder()
    elif encoder_name == 'SeqVec':              # Server Conda env bio_embeddings
        from bio_embeddings.embed import SeqVecEmbedder
        bio_encoder = SeqVecEmbedder()
    elif encoder_name == 'UniRep':
        bio_encoder = UniRepEmbedder()
    elif encoder_name == 'ESM':
        from bio_embeddings.embed import ESMEmbedder
        bio_encoder = ESMEmbedder()
    elif encoder_name == 'ESM1b':
        from bio_embeddings.embed import ESM1bEmbedder
        bio_encoder = ESM1bEmbedder()
    else:
        print(f'Protein encoder {encoder_name} currently not supported.')
        sys.exit()

    embeddings = np.array([])
    with ThreadPoolExecutor(max_workers=8) as executor:
        desc = f'Pre-computing protein encodings with {encoder_name} for {split}: '
        batches = (proteins[i:i + batch_size] for i in range(0, len(proteins), batch_size))
        if encoder_name == 'UniRep':
            threads = [executor.submit(bio_encoder.embed_batch, batch) for batch in batches]
        else:
            threads = [executor.submit(encode_protein, batch, bio_encoder) for batch in batches]
        for t in tqdm(threads, desc=desc, colour='white'):
            emb = t.result()
            embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)

    del bio_encoder
    return embeddings

