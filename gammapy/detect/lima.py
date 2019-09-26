# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
import logging
import numpy as np
from astropy.convolution import Tophat2DKernel
from astropy.coordinates import Angle
from gammapy.stats import significance, significance_on_off

__all__ = ["compute_lima_image", "compute_lima_on_off_image"]

log = logging.getLogger(__name__)

def _setup_kernel(radius, geom):
    radius = Angle(radius).to_value("deg")
    scale = geom.pixel_scales[0].to_value("deg")

    kernel = Tophat2DKernel(radius/scale)
    kernel.normalize("peak")

    return kernel

def compute_lima_image(counts, background, radius="0.1 deg"):
    """Compute Li & Ma significance and flux images for known background.

    Parameters
    ----------
    counts : `~gammapy.maps.WcsNDMap`
        Counts image
    background : `~gammapy.maps.WcsNDMap`
        Background image
    radius : `astropy.coordinates.Angle`
        Correlation radius. Default 0.1 deg.

    Returns
    -------
    images : dict
        Dictionary containing result maps
        Keys are: significance, counts, background and excess

    See Also
    --------
    gammapy.stats.significance
    """
    kernel = _setup_kernel(radius, counts.geom)

    # fft convolution adds numerical noise, to ensure integer results we call
    # np.rint
    counts_conv = np.rint(counts.convolve(kernel.array).data)
    background_conv = background.convolve(kernel.array).data
    excess_conv = counts_conv - background_conv
    significance_conv = significance(counts_conv, background_conv, method="lima")

    return {
        "significance": counts.copy(data=significance_conv),
        "counts": counts.copy(data=counts_conv),
        "background": counts.copy(data=background_conv),
        "excess": counts.copy(data=excess_conv),
    }


def compute_lima_on_off_image(n_on, n_off, a_on, a_off, radius="0.1 deg"):
    """Compute Li & Ma significance and flux images for on-off observations.

    Parameters
    ----------
    n_on : `~gammapy.maps.WcsNDMap`
        Counts image
    n_off : `~gammapy.maps.WcsNDMap`
        Off counts image
    a_on : `~gammapy.maps.WcsNDMap`
        Relative background efficiency in the on region
    a_off : `~gammapy.maps.WcsNDMap`
        Relative background efficiency in the off region
    radius : `astropy.coordinates.Angle`
        Correlation radius. Default 0.1 deg.

    Returns
    -------
    images : dict
        Dictionary containing result maps
        Keys are: significance, n_on, background, excess, alpha

    See Also
    --------
    gammapy.stats.significance_on_off
    """
    kernel = _setup_kernel(radius, n_on.geom)

    # fft convolution adds numerical noise, to ensure integer results we call
    # np.rint
    n_on_conv = np.rint(n_on.convolve(kernel.array).data)
    a_on_conv = a_on.convolve(kernel.array).data

    with np.errstate(invalid="ignore", divide="ignore"):
        alpha_conv = a_on_conv / a_off.data

    significance_conv = significance_on_off(
        n_on_conv, n_off.data, alpha_conv, method="lima"
    )

    with np.errstate(invalid="ignore"):
        background_conv = alpha_conv * n_off.data
    excess_conv = n_on_conv - background_conv

    return {
        "significance": n_on.copy(data=significance_conv),
        "n_on": n_on.copy(data=n_on_conv),
        "background": n_on.copy(data=background_conv),
        "excess": n_on.copy(data=excess_conv),
        "alpha": n_on.copy(data=alpha_conv),
    }
