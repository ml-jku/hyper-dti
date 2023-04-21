
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


def ci_score(y, f):
    """
    Concordance Index from GraphDTA implementation.
    """
    ind = np.argsort(y)
    y = y[ind]
    f = f[ind]
    i = len(y)-1
    j = i-1
    z = 0.0
    S = 0.0
    while i > 0:
        while j >= 0:
            if y[i] > y[j]:
                z = z+1
                u = f[i] - f[j]
                if u > 0:
                    S = S + 1
                elif u == 0:
                    S = S + 0.5
            j = j - 1
        i = i - 1
        j = i-1
    ci = S/z
    return ci


def r_squared_error(y_obs, y_pred):
    """
    Helper function from DeepDTA implementation.
    Reference:
    The original paper by Öztürk et al. (2018) is located at `<https://doi.org/10.1093/bioinformatics/bty593>`.
    """
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    y_pred_mean = [np.mean(y_pred) for y in y_pred]

    mult = sum((y_pred - y_pred_mean) * (y_obs - y_obs_mean))
    mult = mult * mult

    y_obs_sq = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))
    y_pred_sq = sum((y_pred - y_pred_mean) * (y_pred - y_pred_mean) )

    return mult / float(y_obs_sq * y_pred_sq)


def get_k(y_obs, y_pred):
    """
    Helper function from DeepDTA implementation.
    Reference:
    The original paper by Öztürk et al. (2018) is located at `<https://doi.org/10.1093/bioinformatics/bty593>`.
    """
    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)

    return sum(y_obs*y_pred) / float(sum(y_pred*y_pred))


def squared_error_zero(y_obs, y_pred):
    """
    Helper function from DeepDTA implementation.
    Reference:
    The original paper by Öztürk et al. (2018) is located at `<https://doi.org/10.1093/bioinformatics/bty593>`.
    """
    k = get_k(y_obs,y_pred)

    y_obs = np.array(y_obs)
    y_pred = np.array(y_pred)
    y_obs_mean = [np.mean(y_obs) for y in y_obs]
    upp = sum((y_obs - (k*y_pred)) * (y_obs - (k* y_pred)))
    down = sum((y_obs - y_obs_mean)*(y_obs - y_obs_mean))

    return 1 - (upp / float(down))


def rm2_score(ys_orig, ys_line):
    """
    rm2 metric from DeepDTA implementation.
    Reference:
    The original paper by Öztürk et al. (2018) is located at `<https://doi.org/10.1093/bioinformatics/bty593>`.
    """
    r2 = r_squared_error(ys_orig, ys_line)
    r02 = squared_error_zero(ys_orig, ys_line)

    return r2 * (1 - np.sqrt(np.absolute((r2*r2)-(r02*r02))))

