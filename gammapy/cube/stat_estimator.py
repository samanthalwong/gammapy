# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy.optimize import brentq

__all__ = ["StatisticsEstimator"]

class StatisticsEstimator:
    """Compute general statistics (excess significance, asymmetric errors and upper limits
     for any dataset.
     """

    def __init__(self):
        pass

    @staticmethod
    def significance_per_bin(dataset):
        """Compute per bin significance."""
        excess = dataset.excess
        TS0 = dataset.stat_array(np.zeros_like(excess.data))
        TS1 = dataset.stat_array(excess.data)
        significance = np.sign(excess.data) * np.sqrt(TS0 - TS1)
        print(TS1)
        return significance

    @staticmethod
    def significance(dataset):
        """Compute total significance in the full mask (safe & fit)."""
        excess = dataset.excess
        mask = dataset.mask
        TS0 = np.sum(dataset.stat_array(np.zeros_like(excess.data))*mask)
        TS1 = np.sum(dataset.stat_array(excess.data)*mask)
        excess = np.sum(excess.data*mask)
        significance = np.sign(excess) * np.sqrt(TS0 - TS1)
        return significance


