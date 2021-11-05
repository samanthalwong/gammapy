# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from gammapy.maps import Map
from gammapy.datasets import Datasets
from gammapy.modeling.models import PowerLawSpectralModel
from gammapy.stats import CashCountsStatistic, WStatCountsStatistic
from .core import Estimator
from .utils import estimate_exposure_reco_energy

__all__ = [
    "ExcessEstimator",
]

log = logging.getLogger(__name__)


class ExcessEstimator(Estimator):
    """Computes excess, significance and error from a count based dataset.

    If a model is set on the dataset the excess estimator will compute the
    excess taking into account the predicted counts of the model.

    Parameters
    ----------
    n_sigma : float
        Confidence level for the asymmetric errors expressed in number of sigma.
    n_sigma_ul : float
        Confidence level for the upper limits expressed in number of sigma.
    selection_optional : list of str
        Which additional maps to estimate besides delta TS, significance and symmetric error.
        Available options are:

            * "all": all the optional steps are executed
            * "errn-errp": estimate asymmetric errors.
            * "ul": estimate upper limits.

        Default is None so the optionnal steps are not executed.
    energy_edges : `~astropy.units.Quantity`
        Energy edges of the target excess maps bins.
    spectral_model : `~gammapy.modeling.models.SpectralModel`
        Spectral model used for the computation of the flux.
        If None, a Power Law of index 2 is assumed (default).
    """

    tag = "ExcessEstimator"
    _available_selection_optional = ["errn-errp", "ul"]

    def __init__(
        self,
        n_sigma=1,
        n_sigma_ul=2,
        selection_optional=None,
        energy_edges=None,
        spectral_model=None,
    ):
        self.n_sigma = n_sigma
        self.n_sigma_ul = n_sigma_ul
        self.selection_optional = selection_optional
        self.energy_edges = energy_edges

        if spectral_model is None:
            spectral_model = PowerLawSpectralModel(index=2)

        self.spectral_model = spectral_model

    def run(self, datasets):
        """Compute excess, Li & Ma significance and flux maps

        If a model is set on the dataset the excess map estimator will compute the excess taking into account
        the predicted counts of the model.

        Parameters
        ----------
        dataset : `~gammapy.datasets.MapDataset` or `~gammapy.datasets.MapDatasetOnOff`
            input dataset

        Returns
        -------
        images : dict
            Dictionary containing result correlated maps. Keys are:

                * counts : correlated counts map
                * background : correlated background map
                * excess : correlated excess map
                * ts : TS map
                * sqrt_ts : sqrt(delta TS), or Li-Ma significance map
                * err : symmetric error map (from covariance)
                * flux : flux map. An exposure map must be present in the dataset to compute flux map
                * errn : negative error map
                * errp : positive error map
                * ul : upper limit map

        """
        dataset = Datasets(datasets).stack_reduce()
        axis = self._get_energy_axis(dataset)

        resampled_dataset = dataset.resample_energy_axis(
            energy_axis=axis, name=dataset.name
        )
        if dataset.stat_type == "wstat":
            resampled_dataset.models = dataset.models
        elif dataset.stat_type == "cash":
            resampled_dataset.background = dataset.npred().resample_axis(axis=axis)
            resampled_dataset.models = None
        else:
            raise TypeError(f"Unsupported dataset with stat {dataset.stat_sum}")

        result = self.estimate_excess(resampled_dataset)

        return result

    def make_counts_statistics(self, dataset):
        """Build the appropriate CountsStatistic for the dataset."""
        n_on = dataset.counts

        if dataset.stat_type == "wstat":
            n_off = dataset.counts_off
            npred_sig = dataset.npred_signal()
            alpha = dataset.alpha

            return WStatCountsStatistic(
                n_on.data, n_off.data, alpha.data, npred_sig.data
            )
        else:
            npred = dataset.npred()
            return CashCountsStatistic(n_on.data, npred.data)

    def estimate_excess(self, dataset):
        """Estimate excess and ts for a single dataset.

        If exposure is defined, a flux map is also computed.

        Parameters
        ----------
        dataset : `Dataset`
            Dataset
        """
        geom = dataset.counts.geom

        if dataset.mask_safe:
            mask = dataset.mask_safe
        else:
            mask = np.ones(dataset.data_shape, dtype=bool)

        counts_stat = self.make_counts_statistics(dataset)

        n_on = Map.from_geom(geom, data=counts_stat.n_on)
        bkg = Map.from_geom(geom, data=counts_stat.n_on - counts_stat.n_sig)
        excess = Map.from_geom(geom, data=counts_stat.n_sig)

        result = {"counts": n_on, "background": bkg, "excess": excess}

        tsmap = Map.from_geom(geom, data=counts_stat.ts)
        sqrt_ts = Map.from_geom(geom, data=counts_stat.sqrt_ts)
        result.update({"ts": tsmap, "sqrt_ts": sqrt_ts})

        err = Map.from_geom(geom, data=counts_stat.error * self.n_sigma)
        result.update({"err": err})

        if dataset.exposure:
            reco_exposure = estimate_exposure_reco_energy(dataset, self.spectral_model)
            with np.errstate(invalid="ignore", divide="ignore"):
                flux = excess / reco_exposure
                flux.quantity = flux.quantity.to("1 / (cm2 s)").astype(
                    dataset.exposure.data.dtype
                )
        else:
            flux = Map.from_geom(
                geom=dataset.counts.geom, data=np.nan * np.ones(dataset.data_shape)
            )
        result.update({"flux": flux})

        if "errn-errp" in self.selection_optional:
            errn = Map.from_geom(geom, data=counts_stat.compute_errn(self.n_sigma))
            errp = Map.from_geom(geom, data=counts_stat.compute_errp(self.n_sigma))
            result.update({"errn": errn, "errp": errp})

        if "ul" in self.selection_optional:
            ul = Map.from_geom(
                geom, data=counts_stat.compute_upper_limit(self.n_sigma_ul)
            )
            result.update({"ul": ul})

        # return nan values outside mask
        for key in result:
            result[key].data[~mask] = np.nan

        return result
