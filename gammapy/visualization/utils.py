# Licensed under a 3-clause BSD style license - see LICENSE.rst
from gammapy.modeling.models import SkyModel

__all__ = ["plot_spectrum_datasets_off_regions", "plot_spectral_butterfly"]


def plot_spectrum_datasets_off_regions(datasets, ax=None):
    """Plot spectrum datasets of regions.

    Parameters
    ----------
    datasets : list of `SpectrumDatasetOnOff`
        List of spectrum on-off datasets
    """
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches

    ax = plt.gca() or ax

    color_cycle = plt.rcParams["axes.prop_cycle"]
    colors = color_cycle.by_key()["color"]
    handles = []

    for color, dataset in zip(colors, datasets):
        kwargs = {"edgecolor": color, "facecolor": "none"}
        dataset.counts_off.plot_region(ax=ax, **kwargs)

        # create proxy artist for the custom legend
        handle = mpatches.Patch(label=dataset.name, **kwargs)
        handles.append(handle)

    plt.legend(handles=handles)

def plot_spectral_butterfly(
        fit,
        model,
        energy_range,
        ax=None,
        energy_unit="TeV",
        flux_unit="cm-2 s-1 TeV-1",
        energy_power=0,
        n_points=100,
        **kwargs,
    ):
    """Plot spectral model butterfly curve from fit result.

    The errors on the spectral model are obtained from the covariance matrix
    that is accessible from the fit object after fit.covariance is run.

    kwargs are forwarded to `matplotlib.pyplot.plot`

    By default a log-log scaling of the axes is used.

    Parameters
    ----------
    fit : `~gammapy.modeling.Fit`
        The Fit object that performed the spectral fit
    model : `~gammapy.modeling.models`
        The model to plot
    energy_range : `~astropy.units.Quantity`
        Plot range
    ax : `~matplotlib.axes.Axes`, optional
        Axis
    energy_unit : str, `~astropy.units.Unit`, optional
        Unit of the energy axis
    flux_unit : str, `~astropy.units.Unit`, optional
        Unit of the flux axis
    energy_power : int, optional
        Power of energy to multiply flux axis with
    n_points : int, optional
        Number of evaluation nodes

    Returns
    -------
    ax : `~matplotlib.axes.Axes`, optional
        Axis
    """
    if isinstance(model, SkyModel):
        model = model.spectral_model

    if fit._parameters.covariance is None:
        raise ValueError("Run Fit.run() before calling plot_spectral_butterfly")
    covar = fit._parameters.get_subcovariance(model.parameters)
    model.parameters.covariance = covar

    model.plot(
        energy_range=energy_range,
        energy_power=energy_power,
        energy_unit=energy_unit,
        ax=ax,
        flux_unit=flux_unit,
        n_points=n_points,
        **kwargs
    )

    model.plot_error(
        energy_range=energy_range,
        energy_power=energy_power,
        energy_unit=energy_unit,
        ax=ax,
        flux_unit=flux_unit,
        n_points=n_points,
        **kwargs
    )
