# Licensed under a 3-clause BSD style license - see LICENSE.rst
import pytest
import numpy as np
from numpy.testing import assert_allclose
from gammapy.utils.testing import requires_data
from gammapy.cube import ModelEstimator
from gammapy.spectrum import SpectrumDatasetOnOff
from gammapy.modeling import Datasets
from gammapy.modeling.models import PowerLawSpectralModel, SkyModel

@pytest.fixture
def crab_datasets_1d():
    filename = "$GAMMAPY_DATA/joint-crab/spectra/hess/pha_obs23523.fits"
    dataset = SpectrumDatasetOnOff.from_ogip_files(filename)
    datasets = Datasets([dataset])
    return datasets

@pytest.fixture
def PLmodel():
    return PowerLawSpectralModel(amplitude="3e-11 cm-2s-1TeV-1", index=2.7)


@requires_data()
def test_model_estimator_simple(crab_datasets_1d, PLmodel):
    datasets = crab_datasets_1d
    for dataset in datasets:
        dataset.models = SkyModel(spectral_model=PLmodel, name="Crab")

    estimator = ModelEstimator(datasets)

    result = estimator.run(PLmodel, steps=None)

    # First make sure that parameters are correctly set on the dataset.models object
    assert_allclose(datasets[0].models[0].parameters['index'].value, 2.816696962376779)
    assert_allclose(datasets[0].models[0].parameters.error('index'), 0.14957562779580705)
    assert_allclose(datasets[0].models[0].parameters['amplitude'].value, 5.142843823441639e-11)
    assert_allclose(datasets[0].models[0].parameters.error('amplitude'), 6.42467002840316e-12)

@requires_data()
def test_model_estimator_ts(crab_datasets_1d, PLmodel):
    datasets = crab_datasets_1d
    for dataset in datasets:
        dataset.models = SkyModel(spectral_model=PLmodel, name="Crab")

    estimator = ModelEstimator(datasets)

    result = estimator.run(PLmodel, steps=None)
