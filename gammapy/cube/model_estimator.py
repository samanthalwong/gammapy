# Licensed under a 3-clause BSD style license - see LICENSE.rst
import logging
import numpy as np
from itertools import combinations
import astropy.units as u
from gammapy.modeling import Fit, Datasets

__all__ = ["ModelEstimator"]

log = logging.getLogger(__name__)


class ModelEstimator:
    """Model parameters estimator.

    Estimates free parameters of a given model component for a group of datasets

    Parameters
    ----------
    sigma : int
        Sigma to use for asymmetric error computation.
    sigma_ul : int
        Sigma to use for upper limit computation.
    reoptimize : bool
        Re-optimize other free model parameters. Default is True.
    n_scan_values : int
        Number of values used to scan fit stat profile
    scan_n_err : float
        Range to scan in number of parameter error
    """

    def __init__(
            self,
            sigma=1,
            sigma_ul=2,
            reoptimize=True,
            n_scan_values=30,
            scan_n_err=3
    ):
        self.sigma = sigma
        self.sigma_ul = sigma_ul
        self.reoptimize = reoptimize
        self.n_scan_values = n_scan_values
        self.scan_n_err = scan_n_err

    def __str__(self):
        s = f"{self.__class__.__name__}:\n"
        return s

    def _check_datasets(self, datasets):
        """Check datasets geometry consistency and return Datasets object"""
        if not isinstance(datasets, Datasets):
            datasets = Datasets(datasets)

        if not datasets.is_all_same_type and datasets.is_all_same_shape:
            raise ValueError(
                "Flux estimation requires a list of datasets"
                " of the same type and data shape."
            )
        return datasets

    def _freeze_parameters(self, datasets, model):
        """Freeze all other parameters"""
        # freeze other parameters
        for par in datasets.parameters:
            if par not in model.parameters:
                par.frozen = True

    def _compute_scan_values(self, value, value_error, par_min, par_max):
        """Define parameter value range to be scanned"""
        min_range = value - self.scan_n_err * value_error
        if not np.isnan(par_min):
            min_range = np.maximum(par_min, min_range)
        max_range = value + self.scan_n_err * value_error
        if not np.isnan(par_max):
            max_range = np.minimum(par_max, max_range)

        return np.linspace(min_range, max_range, self.n_scan_values)

    def run(self, datasets, model, steps="all"):
        """Run the model estimator.

        Parameters
        ----------
        datasets : `~gammapy.modeling.Datasets` or a list of datasets
            Input datasets.
        model : `~gammapy.modeling.SkyModel`
            The model to estimate
        steps : list of str
            Which steps to execute. Available options are:

                * "errn-errp": estimate asymmetric errors.
                * "ul": estimate upper limits.
                * "stat_profile": estimate fit statistic profiles.

            By default all steps are executed.

        Returns
        -------
        result : `~astropy.table.Table`
            Estimated flux and errors.
        """
        datasets = self._check_datasets(datasets)

        if steps == "all":
            steps = ["errp-errn", "ul", "stat_profile"]

        self.fit = Fit(datasets)

        self.fit_result = self.estimate_best_fit()
        covar = self.fit_result.parameters.get_subcovariance(model.parameters)
        model.parameters.covariance = covar

        params = [_ for _ in model.parameters if _.frozen is False]

        for par in params:
            result = {"value": par.value}
            result.update({"error": self.fit_result.parameters.error(par)})
            #print(self.estimate_parameter(par, steps))
            print(result)
        return result

    def estimate_best_fit(self):
        # Find best fit solution
        result = self.fit.optimize()

        if not result.success:
            log.warning("Fit failed for model estimate, stopping model estimation.")
            return result
        self.fit_res = result
        result = self.fit.covariance()

        return result

    def estimate_ts(self, parameter=None, null_value=0):
        """Estimate of null hypothesis vs best fit.

        Null hypothesis is given by parameter.value=null_value.

        By default search for amplitude or norm parameter and use 0 as null value.

        Parameters
        ----------
        parameter : `~gammapy.modeling.Parameter`
            The parameter defining the null hypothesis
        null_value : float
            The null value
        """
        if self.fit_result is None:
            self.estimate_best_fit()
        if parameter is None:
            if 'amplitude' in self.fit_result.parameters.names:
                index = self.fit_result.parameters.names.index("amplitude")
            elif 'norm' in self.fit_result.parameters.names:
                index = self.fit_result.parameters.names.index("norm")
            else:
                raise ValueError("Cannot find parameter for TS estimation.")
            parameter = self.fit_result.parameters[index]

        with datasets.parameters.restore_values:
            parameter.value = null_value
            parameter.frozen = True
            result = self.fit.optimize()
            return result.total_stat - self.fit_res.total_stat

    def estimate_parameter(self, parameter, steps):
        """Fit norm of the flux point.

        Parameters
        ----------
        parameter : `~gammapy.modeling.Parameter`
            the parameter to be estimated

        Returns
        -------
        result : dict
            Dict with "norm" and "stat" for the flux point.
        """
        self.estimate_best_fit()
        value_err = self.fit_result.parameters.error(parameter)
        result = {}
        if "errp-errn" in steps:
            res = self.fit.confidence(parameter=parameter.name, sigma=self.sigma)
            result.update({"errp": res["errp"], "errn": res["errn"]})

        if "ul" in steps:
            res = self.fit.confidence(parameter=parameter, sigma=self.sigma_ul)
            result.update({"ul": res["errp"] + parameter.value})

        if "stat_profile" in steps:
            min_range = parameter.value - self.scan_n_err * value_err
            if not np.isnan(parameter.min):
                min_range = np.maximum(parameter.min, min_range)
            max_range = parameter.value + self.scan_n_err * value_err
            if not np.isnan(parameter.max):
                max_range = np.minimum(parameter.max, max_range)

            param_values = self._compute_scan_values(
                parameter.value,
                value_err,
                parameter.min,
                parameter.max
            )
            res = self.fit.stat_profile(
                parameter, values=param_values, reoptimize=self.reoptimize
            )
            result.update({"values": res["values"], "stat": res["stat"]})
        return result

    def make_contours(self, parameters="all", npoints=10, sigma=1):
        """Make contours."""
        if parameters == "all":
            parameters = [_ for _ in self.fit_result.parameters if _.frozen is False]
        else:
            parameters = [self.fit_result.parameters[name] for name in parameters]

        results = {}
        for param1, param2 in combinations(parameters, r=2):
            name = f"{param1.name}_{param2.name}"
            res = self.fit.minos_contour(param1, param2, npoints, sigma)
            par1_contour = u.Quantity(res['x'], param1.unit)
            par2_contour = u.Quantity(res['y'], param2.unit)
            results[name] = {param1.name: par1_contour, param2.name: par2_contour}
        return results


