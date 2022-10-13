import warnings
import numpy as np
from sklearn.metrics import matthews_corrcoef


def mcc_score(y_true, y_prob, threshold: float):
    """
    Matthews Correlation Coefficient can be computed in three different ways,
    1. for training computed on threshold 0.5
    2. for validation computed for 0.5 but also includes finding optimal threshold on full set.
    3. for testing computed for optimal validation threshold.
    Author: Emma Svensson
    """

    # For Threshold: None find optimal threshold
    best_threshold = None
    if threshold is None:
        score = None
        best_score = 0.0
        for tau in np.linspace(0.1, 0.9, 81):
            y_pred = (y_prob > tau).tolist()
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", category=RuntimeWarning)
                tmp_score = matthews_corrcoef(y_true=y_true, y_pred=y_pred)

            # Always return the unbiased score for threshold 0.5 if threshold was None
            if tau == 0.5:
                score = tmp_score

            if tmp_score > best_score:
                best_score = tmp_score
                best_threshold = tau
    else:
        y_pred = (y_prob > threshold).tolist()
        score = matthews_corrcoef(y_true=y_true, y_pred=y_pred)

    return score, best_threshold

