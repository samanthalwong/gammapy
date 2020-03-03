.. include:: ../references.txt

.. _datasets:

***************************
datasets - Reduced datasets
***************************

.. currentmodule:: gammapy.datasets

Introduction
============

The `gammapy.datasets` sub-package contains classes to handle reduced
gamma-ray data for modeling and fitting.

Getting Started
===============

The `Dataset` class bundles reduced data, IRFs and model to perform
likelihood fitting and joint-likelihood fitting.

Datasets in Gammapy contain reduced data, models, and the likelihood function
fit statistic for a given set of model parameters. All datasets contain a
`~gammapy.modeling.models.Models` container with one or more
`~gammapy.modeling.models.SkyModel` objects that represent additive emission
components.

To model and fit data in Gammapy, you have to create a
`~gammapy.modeling.Datasets` container object with one or multiple
`~gammapy.modeling.Dataset` objects. Gammapy has built-in support to create and
analyse the following datasets: `~gammapy.cube.MapDataset`,
`~gammapy.cube.MapDatasetOnOff`, `~gammapy.spectrum.SpectrumDataset`,
`~gammapy.spectrum.SpectrumDatasetOnOff` and
`~gammapy.spectrum.FluxPointsDataset`.


MapDataset
==========

The map datasets represent 3D cubes (`~gammapy.maps.WcsNDMap` objects) with two
spatial and one energy axis. For 2D images the same map objects and map datasets
are used, an energy axis is present but only has one energy bin. The
`~gammapy.cube.MapDataset` contains a counzts map, background is modeled with a
`~gammapy.modeling.models.BackgroundModel`, and the fit statistic used is
``cash``. The `~gammapy.cube.MapDatasetOnOff` contains on and off count maps,
background is implicitly modeled via the off counts map, and the ``wstat`` fit
statistic.


SpectrumDataset
===============

The spectrum datasets represent 1D spectra (`~gammapy.spectrum.CountsSpectrum`
objects) with an energy axis. There are no spatial axes or information, the 1D
spectra are obtained for a given on region. The
`~gammapy.spectrum.SpectrumDataset` contains a counts spectrum, background is
modeled with a `~gammapy.spectrum.CountsSpectrum`, and the fit statistic used is
``cash``. The `~gammapy.spectrum.SpectrumDatasetOnOff` contains on on and off
count spectra, background is implicitly modeled via the off counts spectrum, and
the ``wstat`` fit statistic.

FluxPointsDataset
=================

This dataset is made for flux measurements. It does not contain any IRF nor counts information.
The `~gammapy.spectrum.FluxPointsDataset` contains flux values in the form of a
`~gammapy.spectrum.FluxPoints` and a spectral model. The fit statistic used here is
``chi2``.


Reference/API
=============

.. automodapi:: gammapy.datasets
    :no-inheritance-diagram:
    :include-all-objects:
