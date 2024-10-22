# Licensed under a 3-clause BSD style license - see LICENSE.rst
"""Utility functions for the high level interface API."""

from astropy.table import Table
from gammapy.utils.scripts import make_path


def make_obs_table_selection(obs_table, obs_ids=[], obs_file=None, obs_cone=None):
    """Return list of obs_ids after filtering on observation table.

    Parameters
    ----------
    obs_table : `~gammapy.data.ObservationTable`
        Input observation table.
    obs_ids : list[str]
        List of obsids. Default is [].
    obs_file : str, optional
        observation list file name. Default is None.
    obs_cone : dict, optional
        dictionary defining observation cone. Default is None.
    Returns
    -------
    selected_obs_ids : list[int]
        list of selected obs_ids
    """
    # Reject configs with list of obs_ids and obs_file set at the same time
    if len(obs_ids) and obs_file is not None:
        raise ValueError(
            "Values for both parameters obs_ids and obs_file are not accepted."
        )

    # First select input list of observations from obs_table
    if len(obs_ids):
        selected_obs_table = obs_table.select_obs_id(obs_ids)
    elif obs_file is not None:
        path = make_path(obs_file)
        ids = list(Table.read(path, format="ascii", data_start=0).columns[0])
        selected_obs_table = obs_table.select_obs_id(ids)
    else:
        selected_obs_table = obs_table

    # Apply cone selection
    if obs_cone.lon is not None:
        cone = dict(
            type="sky_circle",
            frame=obs_cone.frame,
            lon=obs_cone.lon,
            lat=obs_cone.lat,
            radius=obs_cone.radius,
            border="0 deg",
        )
        selected_obs_table = selected_obs_table.select_observations(cone)

    return selected_obs_table["OBS_ID"].tolist()
