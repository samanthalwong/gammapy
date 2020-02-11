# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from scipy.optimize import brentq
from gammapy.stats import cash, wstat


__all__ = ["StatisticsEstimator"]

class StatisticsEstimator:
    """Compute general statistics (excess significance, asymmetric errors and upper limits
     for any dataset.
     """

    def __init__(self, sigma=1, sigma_ul=3, method='brentq'):
        """Estimate statistics """
        self.sigma = sigma
        self.sigma_ul = sigma_ul
        self.method = method

    @staticmethod
    def significance_per_bin(dataset):
        """Compute per bin significance."""
        excess = dataset.excess
        TS0 = dataset.stat_array(np.zeros_like(excess.data))
        TS1 = dataset.stat_array(excess.data)
        significance = np.sign(excess.data) * np.sqrt(TS0 - TS1)
        return significance

    @staticmethod
    def significance(dataset):
        """Compute total significance in the full mask (safe & fit)."""
        excess = dataset.excess
        mask = dataset.mask
        TS0 = dataset.stat_sum(np.zeros_like(excess.data))
        TS1 = dataset.stat_sum(excess.data)
        excess = np.sum(excess.data*mask)
        significance = np.sign(excess) * np.sqrt(TS0 - TS1)
        return significance

    def run(self, dataset, steps="all"):
        if steps is "all":
            steps = ["errp-errn", "ul"]

        excess = np.sum(dataset.excess.data*dataset.mask)
        significance = self.significance(dataset)


        if "errp-errn" in steps:
            errn = wstat_compute_errn(n_on, n_off, alpha)
            errp = wstat_compute_errp(n_on, n_off, alpha)
        if "ul" in steps:
            ul = wstat_compute_upper_limit(n_on, n_off, alpha)

        return {
            'excess': excess,
            'significance': significance,
            'errn': errn,
            'errp': errp,
            'ul': ul
        }

class WStat:
    def __init__(self, n_on, n_off, alpha):
        self.n_on = n_on
        self.n_off = n_off
        self.alpha = alpha

    @property
    def excess(self):
        return self.n_on - self.alpha * self.n_off

    @property
    def std(self):
        return np.sqrt(self.n_on + self.alpha ** 2 * self.n_off)

    @property
    def TS_null(self):
        return wstat(self.n_on, self.n_off, self.alpha, 0)

    @property
    def TS_max(self):
        return wstat(self.n_on, self.n_off, self.alpha, self.excess)

    @property
    def significance(self):
        return np.sign(self.excess)*np.sqrt(self.TS_null-self.TS_max)

    @staticmethod
    def _wstat_fcn(self, mu, delta):
        return wstat(self.n_on, self.n_off, self.alpha, mu) - delta

    def compute_errn(self, n_sigma=1):
        min_range = self.excess - 2 * n_sigma * self.std

        errn = brentq(
            self._wstat_fcn,
            min_range,
            self.excess,
            args=(self.TS_max + n_sigma))

        return errn - self.excess

    def compute_errp(self, n_sigma=1):
        max_range = self.excess + 2 * n_sigma * self.std

        errp = brentq(
            self._wstat_fcn,
            self.excess,
            max_range,
            args=(self.TS_max + n_sigma))

        return errp - self.excess

    def compute_upper_limit(self, n_sigma=3):
        min_range = np.maximum(0, self.excess)
        max_range = min_range + 2 * n_sigma * self.std
        return brentq(
            self._wstat_fcn,
            min_range,
            max_range,
            args=(self.TS_max + n_sigma))

class Cash:
    def __init__(self, n_on, mu_bkg):
        self.n_on = n_on
        self.mu_bkg= mu_bkg

    @property
    def excess(self):
        return self.n_on - self.mu_bkg

    @property
    def std(self):
        return np.sqrt(self.n_on+1)

    @property
    def TS_null(self):
        return cash(self.n_on, self.mu_bkg + 0)

    @property
    def TS_max(self):
        return wstat(self.n_on, self.n_on)

    @property
    def significance(self):
        return np.sign(self.excess)*np.sqrt(self.TS_null-self.TS_max)

    @staticmethod
    def _cash_fcn(self, mu, delta):
        return cash(self.n_on, self.mu_bkg + mu) - delta

    def compute_errn(self, n_sigma=1):
        min_range = self.excess - 2 * n_sigma * self.std

        errn = brentq(
            self._cash_fcn,
            min_range,
            self.excess,
            args=(self.TS_max + n_sigma))

        return errn - self.excess

    def compute_errp(self, n_sigma=1):
        max_range = self.excess + 2 * n_sigma * self.std

        errp = brentq(
            self._cash_fcn,
            self.excess,
            max_range,
            args=(self.TS_max + n_sigma))

        return errp - self.excess

    def compute_upper_limit(self, n_sigma=3):
        min_range = np.maximum(0, self.excess)
        max_range = min_range + 2 * n_sigma * self.std
        return brentq(
            self._cash_fcn,
            min_range,
            max_range,
            args=(self.TS_max + n_sigma))


