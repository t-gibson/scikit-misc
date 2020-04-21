import numpy as np

from sklearn.processing import FunctionTransformer


def fct_lump(x, pct=0.8, other_level="Other"):
    """
    Python implementation of https://forcats.tidyverse.org/reference/fct_lump.html
    
    Args:
        x (np.narray like)
        pct (float): minimum percentage of the data that must be covered.
    """
    if not 0.0 < pct < 1.0:
        raise ValueError(f"pct must be between 0 and 1. Instead it was {pct}")

    categories, counts = np.unique(x, return_counts=True)
    n = counts.sum()
    descending_order = np.flip(np.argsort(counts))
    
    pct_covered = np.cumsum(counts[descending_order]) / n
    index_covered = np.argmax(pct_covered > pct)
    
    ctg_to_keep = categories[descending_order][:index_covered]
    translation = {ctg: ctg if ctg in ctg_to_keep else other_level for ctg in categories}
    
    return np.array([translation[val] for val in x])


def fct_lump_cols(X, *args):
    """
    Apply fct_lump across an entire array.
    
    Args:
        X (np.ndarray): An array of features treated as categorical.
        **args: passed onto fct_lump()
    """
    return np.apply_along_axis(fct_lump, 0, X, *args)


FctLumpTransformer = FunctionTransformer(fct_lump_cols, check_inverse=False, validate=False)