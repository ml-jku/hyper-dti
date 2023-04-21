
import sklearn.metrics as metrics
from torch.nn import BCEWithLogitsLoss, MSELoss, L1Loss
from hyper_dti.utils.metrics import mcc_score, ci_score, rm2_score


METRICS = {
    'classification': {
        'AUC': metrics.roc_auc_score,
        'AUPRC': metrics.average_precision_score,
        'MCC': mcc_score,
    },
    'regression': {
        'MSE': metrics.mean_squared_error,
        'MAE': metrics.mean_absolute_error,
        'CI': ci_score,
        'rm2': rm2_score
    }
}

MAX_FOLDS = {
    'Lenselink': 9,
    'KIBA': 4,
    'Davis': 4
}

BIOACTIVITY_THRESHOLD = {
    'Lenselink': 6.5,
    'KIBA': 12.1,
    'Davis': 7
}

# Lenselink
EXCLUDED = ['CHEMBL6165']
NUM_CLASSES = 1
NUM_TASKS = 1226
TEMPORAL_SPLIT = [2012, 2013]

# Drug embedding
DRUG_LATENT_DIM = {'MolBert': 1536, 'CDDD': 512}
MAIN_BATCH_SIZE = -1
OVERSAMPLING = 32

# Target embedding
TARGET_LATENT_DIM = {'UniRep': 1900, 'SeqVec': 1024,
                      'ProtBert': 1024, 'ProtT5': 1024,
                      'ESM1b': 1280, 'ESM2': 2560}
TARGET_CHARS = {
    "A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "H": 8, "I": 9, "K": 10, "L": 11, "M": 12,
    "N": 13, "P": 14, "Q": 15, "R": 16, "S": 17, "T": 18, "V": 19, "W": 20, "Y": 21, "X": 22
}
MAX_TARGET_LEN = 1333


LOSS_CLASS = {
    'classification': ['BCE'],
    'regression': ['MSE', 'MAE']
}

LOSS_FUNCTION = {
    'BCE': BCEWithLogitsLoss,
    'MSE': MSELoss,
    'MAE': L1Loss,
}
