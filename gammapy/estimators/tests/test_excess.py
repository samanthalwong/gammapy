# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
import astropy.units as u
from gammapy.datasets import SpectrumDatasetOnOff
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
def test_spectrum_dataset():
    dataset = SpectrumDatasetOnOff.read("$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits")

    ee = ExcessEstimator()
    excess = ee.run(dataset)
    print(excess["excess"].data)
    assert False