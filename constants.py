import sklearn.metrics as metrics
from utils.metrics import mcc_score

METRICS = {
    'AUC': metrics.roc_auc_score,
    'MCC': mcc_score,
}

# ChEMBL
EXCLUDED = ['CHEMBL6165']
NUM_CLASSES = 1
NUM_TASKS = 1226
BIOACTIVITY_THRESHOLD = 6.5
TEMPORAL_SPLIT = [2012, 2013]

# Molecule Embedding
MOLECULE_IN_DIM = 100      # TODO
MOLECULE_LATENT_DIM = {'MolBert': 1536, 'CDDD': 512, 'ECFP': 2048}
MAIN_BATCH_SIZE = -1
OVERSAMPLING = 32

# Protein Embedding
PROTEIN_IN_DIM = 100        # TODO
PROTEIN_LATENT_DIM = {'UniRep': 1900, 'SeqVec': 1024,
                      'ProtTransBertBFD': 1024, 'ProtTransT5XLU50': 1024,
                      'ESM': 1280, 'ESM1b': 1280}
PROTEIN_CHARS = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "K": 10, "L": 11, "M": 12,
    "N": 13, "P": 14, "Q": 15, "R": 16, "S": 17, "T": 18, "V": 19, "W": 20, "Y": 21, "X": 22
}
MAX_PROTEIN_LEN = 1333


LOSS_CLASS = {'classification': ['bce'],
              'regression': ['mse', 'mae']}
