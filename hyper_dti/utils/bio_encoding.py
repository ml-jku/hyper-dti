
import os
import gc
import sys
import torch
import numpy as np
from tqdm import tqdm
from itertools import tee
from typing import Any, Dict, List, Generator, Iterable
from concurrent.futures import ThreadPoolExecutor


def encode_drug(batch, encoder, name):
    """ Wraps encoding of drug compounds, i.e. drugs, from different encoders. """
    embeddings = encoder(batch)
    return embeddings if name == 'CDDD' else embeddings[0]


def encode_target(batch, encoder):
    """ Wraps encoding of protein targets, i.e. targets, from different encoders. """
    full_embeddings = encoder.embed_batch(batch)
    embeddings = np.array([])
    for emb in full_embeddings:
        emb = encoder.reduce_per_target(emb)
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
    def reduce_per_target(embedding):
        return embedding.mean(axis=0)


class ESM2Embedder:
    """
    Wrapper for ESM-2 model inspired by bio-embedding implementation of older esm models.
    """

    name = 'esm2'
    embedding_dimension = 2560
    necessary_files = ["model_file"]
    max_len = 1022

    _picked_layer = 36

    def __init__(self):
        super().__init__()

        import esm

        model, alphabet = esm.pretrained.esm2_t36_3B_UR50D()
        model.eval()
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._model = model.to(self._device)
        self._batch_converter = alphabet.get_batch_converter()

        param_size = 0
        for param in model.parameters():
            param_size += param.nelement() * param.element_size()
        buffer_size = 0
        for buffer in model.buffers():
            buffer_size += buffer.nelement() * buffer.element_size()

        size_all_mb = (param_size + buffer_size) / 1024 ** 2
        print('DEBUG: Model size: {:.3f}MB'.format(size_all_mb))

    def embed(self, sequence: str) -> np.ndarray:
        [embedding] = self.embed_batch([sequence])
        return embedding

    def embed_batch(self, batch: List[str]) -> Generator[np.ndarray, None, None]:
        batch, batch_copy = tee(batch)
        data = [(str(pos), sequence[:min(len(sequence), ESM2Embedder.max_len)]) for pos, sequence in enumerate(batch)]

        """
        data = []
        pos = 0
        target_ind = []
        for target, sequence in enumerate(batch):
            sequence_batch = []
            for i in range(len(sequence) // 1022 + 1): 
                sequence_batch.append(sequence[i * 1022, min(len(sequence), (i+1) * 1022)])

            for s in sequence_batch:
                data.append((str(pos), s))
                pos += 1
                target_ind.append(target)
        """

        batch_labels, batch_strs, batch_tokens = self._batch_converter(data)

        with torch.no_grad():
            results = self._model(
                batch_tokens.to(self._device), repr_layers=[self._picked_layer]
            )
        token_embeddings = results["representations"][self._picked_layer]

        # Generate per-sequence embeddings via averaging
        # NOTE: token 0 is always a beginning-of-sequence token, so the first residue is token 1.
        for i, (_, seq) in enumerate(data):
            yield token_embeddings[i, 1: len(seq) + 1].cpu().numpy()

    def _assert_max_len(self, sequences: Iterable[str]):
        max_len = max((len(i) for i in sequences), default=0)
        if max_len > self.max_len:
            raise ValueError(
                f"{self.name} only allows sequences up to {self.max_len} residues, "
                f"but your longest sequence is {max_len} residues long"
            )

    @staticmethod
    def reduce_per_target(embedding: np.ndarray) -> np.ndarray:
        return embedding.mean(0)


def precompute_drug_embeddings(drugs, encoder_name, split, batch_size):
    gc.collect()

    if encoder_name == 'CDDD':          # Server conda env cddd not work on server
        from cddd.inference import InferenceModel
        CDDD_MODEL_DIR = 'checkpoints/CDDD/default_model'
        assert os.path.exists(CDDD_MODEL_DIR), \
            'Error: default model should be downloaded according to CDDD github.' \
            'Place the default_model folder under checkpoints/CDDD/'
        cddd_model = InferenceModel(CDDD_MODEL_DIR)
        mol_encoder = cddd_model.seq_to_emb
    elif encoder_name == 'MolBert':     # Server conda env molbert
        from molbert.utils.featurizer.molbert_featurizer import MolBertFeaturizer
        MOLBERT_MODEL_DIR = 'checkpoints/MolBert/molbert_100epochs/checkpoints/last.ckpt'
        assert os.path.exists(MOLBERT_MODEL_DIR), \
            'Error: checkpoint should be downloaded according to CDDD github.' \
            'Place the file last.ckpt under checkpoints/MolBert/molbert_100epochs/checkpoints/last.ckpt'
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
        print(f'Target encoder {encoder_name} currently not supported.')
        sys.exit()

    embeddings = np.array([])
    with ThreadPoolExecutor(max_workers=8) as executor:
        desc = f'Pre-computing drug encodings with {encoder_name} for {split}: '
        batches = (drugs[i:i + batch_size] for i in range(0, len(drugs), batch_size))
        threads = [executor.submit(encode_drug, batch, mol_encoder, encoder_name) for batch in batches]
        for t in tqdm(threads, desc=desc, colour='white'):
            emb = t.result()
            embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)
    return embeddings


def precompute_target_embeddings(targets, encoder_name, batch_size, n_jobs=8):
    gc.collect()

    if encoder_name == 'ProtBert':  # Server Conda env bio_embeddings
        from bio_embeddings.embed import ProtTransBertBFDEmbedder
        bio_encoder = ProtTransBertBFDEmbedder()
    elif encoder_name == 'ProtT5':  # Server Conda env bio_embeddings
        from bio_embeddings.embed import ProtTransT5XLU50Embedder
        bio_encoder = ProtTransT5XLU50Embedder()
    elif encoder_name == 'SeqVec':  # Server Conda env bio_embeddings. local requires batch_size 4
        from bio_embeddings.embed import SeqVecEmbedder
        bio_encoder = SeqVecEmbedder()
    elif encoder_name == 'UniRep':
        bio_encoder = UniRepEmbedder()
    elif encoder_name == 'ESM2':
        bio_encoder = ESM2Embedder()
    elif encoder_name == 'ESM1b':
        from bio_embeddings.embed import ESM1bEmbedder
        bio_encoder = ESM1bEmbedder()
    else:
        print(f'Target encoder {encoder_name} currently not supported.')
        sys.exit()

    embeddings = np.array([])

    desc = f'Pre-computing target encodings with {encoder_name}: '
    batches = (targets[i:i + batch_size] for i in range(0, len(targets), batch_size))
    if n_jobs > 0:
        assert n_jobs < 9, f'{n_jobs} are too many workers for parallel computing.'
        with ThreadPoolExecutor(max_workers=n_jobs) as executor:
            if encoder_name == 'UniRep':
                threads = [executor.submit(bio_encoder.embed_batch, batch) for batch in batches]
            else:
                threads = [executor.submit(encode_target, batch, bio_encoder) for batch in batches]
            for t in tqdm(threads, desc=desc, colour='white'):
                emb = t.result()
                embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)
    else:
        results = [encode_target(batch, bio_encoder) for batch in batches]
        for emb in tqdm(results, desc=desc, colour='white'):
            embeddings = emb if len(embeddings) == 0 else np.append(embeddings, emb, axis=0)

    del bio_encoder
    return embeddings

