import numpy as np

def ML_estimator_s2(s2, df):
    """Max-likelihood estimator for variance from a sample of different-Ndf
    estimates.

    The max-likelihood estimator is simply the weighted average with the
    number of degrees fo freedom as weights.

    Paramters
    ---------
    s2 : array_like
        Unbiased array estimators.
    df : array_like
        The number of degrees of freedom of each estimate in `s2`.
    """
    return np.average(s2, weights=df)


def ML_estimator_ds2(s2, df):
    """Uncertainty of the max-likelihood estimator `ML_estimator_s2`.

    Paramters
    ---------
    s2 : array_like
        Unbiased array estimators.
    df : array_like
        The number of degrees of freedom of each estimate in `s2`.
    """
    s2_ml = ML_estimator_s2(s2, df)
    denom = np.sum(df * (2.0 * s2 / s2_ml - 1.0))
    return s2_ml * np.sqrt(2.0 / denom)
