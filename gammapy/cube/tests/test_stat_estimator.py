# Licensed under a 3-clause BSD style license - see LICENSE.rst
import numpy as np
from numpy.testing import assert_allclose
from gammapy.maps import MapAxis
from gammapy.spectrum import SpectrumDataset, SpectrumDatasetOnOff
from gammapy.cube.stat_estimator import StatisticsEstimator


class TestSpectrumDataset:
    def setup(self):
        e_reco = MapAxis.from_energy_bounds(1.,10.,3, unit='TeV')
        self.dataset = SpectrumDataset.create(e_reco.edges)
        self.dataset.counts += 2
        self.dataset.background += 1
        self.dataset.mask_safe = np.ones_like(self.dataset.counts.data, dtype='bool')

    def test_significance_per_bin(self):
        significance = StatisticsEstimator.significance_per_bin(self.dataset)

        assert len(significance) == 3
        assert_allclose(significance, 0.87897026)

    def test_significance(self):
        significance = StatisticsEstimator.significance(self.dataset)

        self.dataset.mask_fit = np.zeros_like(self.dataset.counts.data, dtype='bool')
        self.dataset.mask_fit[0] = True
        significance_mask = StatisticsEstimator.significance(self.dataset)

        assert_allclose(significance, 1.52242115)
        assert_allclose(significance_mask, 0.87897026)

class TestSpectrumDatasetOnOff:
    def setup(self):
        e_reco = MapAxis.from_energy_bounds(1.,10.,3, unit='TeV')
        self.dataset = SpectrumDatasetOnOff.create(e_reco.edges)
        self.dataset.counts += 2
        self.dataset.counts_off += 2
        self.dataset.acceptance[:] =1
        self.dataset.acceptance_off[:] = 2

        self.dataset.mask_safe = np.ones_like(self.dataset.counts.data, dtype='bool')

    def test_significance_per_bin(self):
        significance = StatisticsEstimator.significance_per_bin(self.dataset)

        assert len(significance) == 3
        assert_allclose(significance, 0.6863906632709494)

    def test_significance(self):
        significance = StatisticsEstimator.significance(self.dataset)

        self.dataset.mask_fit = np.zeros_like(self.dataset.counts.data, dtype='bool')
        self.dataset.mask_fit[0] = True
        significance_mask = StatisticsEstimator.significance(self.dataset)

        assert_allclose(significance, 1.1888635026261853)
        assert_allclose(significance_mask, 0.6863906632709494)
