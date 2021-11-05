# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.datasets import SpectrumDatasetOnOff, Datasets
from gammapy.estimators import ExcessEstimator
from gammapy.maps import Map, MapAxis, WcsGeom
from gammapy.irf import PSFMap
from gammapy.modeling.models import (
    GaussianSpatialModel,
    PowerLawSpectralModel,
    SkyModel,
)
from gammapy.utils.testing import requires_data
from gammapy.estimators.utils import estimate_exposure_reco_energy



@requires_data()
def test_excess_spectrum_dataset():

    dataset1 = SpectrumDatasetOnOff.read("$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits")
    dataset2 = SpectrumDatasetOnOff.read("$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23526.fits")

    datasets = Datasets([dataset1, dataset2])

    edges = dataset1.counts.geom.axes["energy"].downsample(8).edges

    ee = ExcessEstimator(energy_edges=edges, selection_optional="all")
    excess = ee.run(datasets)
    assert np.all(np.isnan(excess["excess"].data[:3]))
    assert_allclose(np.squeeze(excess["excess"].data[4:]), [56., 111.75, 51.8333, 12.3333, 1.8333, 0.], atol=1e-4)
    assert_allclose(np.squeeze(excess["err"].data[4:]), [7.767453, 10.9363,  7.5064,  3.6132,  1.4191,  0.], atol=1e-4)
    assert_allclose(np.squeeze(excess["ul"].data[4:]), [72.8891, 134.9694, 68.2006, 20.9458, 6.1302, 1.818989e-12], atol=1e-4)


