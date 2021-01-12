# Licensed under a 3-clause BSD style license - see LICENSE.rst

__all__ = ["FluxPointsCollection"]


class FluxPointsCollection:
    """A generalized FluxPoints object.

    A FluxPointsCollection represents a series of flux points.
    FluxPoints can be produced in different time intervals, for different sources
    or in different regions.

    Flux is internally represented in the form of a `~gammapy.estimators.FluxEstimate`
    which follows the likelihood SED type with 'norm' quantities and a reference spectral model.

    Parameters
    ----------
    data : `~astropy.table.Table`
        Table with flux point data
    reference_model : `~gammapy.modeling.models.SpectralModel`
        The reference spectral model
    """
    def __init__(self, data, reference_model):
        self.data = data
        self.reference_model = reference_model



        #TODO: add inheritance from FluxEstimate

    @staticmethod
    def _validate_table(table):
        """Validate input table."""
        required = set(["e_min", "e_max", "e_ref", "norm"])

        if not required.issubset(table.colnames):
            missing = required.difference(table.colnames)
            raise ValueError(f"Missing columns {missing})")
