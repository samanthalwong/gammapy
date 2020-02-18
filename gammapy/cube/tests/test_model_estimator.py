# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.utils.testing import requires_data
from gammapy.cube import ModelEstimator
from gammapy.spectrum import SpectrumDatasetOnOff
from gammapy.modeling import Datasets
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel


@requires_data()
def test_model_estimator_simple():
    filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    dataset = SpectrumDatasetOnOff.from_ogip_files(filename)
    datasets = Datasets([dataset])

    PLmodel = PowerLawSpectralModel(amplitude="3e-11 cm-2s-1TeV-1", index=2.7)
    dataset.models = SkyModel(spectral_model=PLmodel, name="Crab")

    estimator = ModelEstimator()

    result = estimator.run(datasets, PLmodel, steps=["errp-errn"])

    assert_allclose(result['index']['value'],2.816898113390892)
    assert 1==0
