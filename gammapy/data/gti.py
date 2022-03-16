# Licensed under a 3-clause BSD style license - see LICENSE.rst
import copy
from operator import le, lt
import numpy as np
from astropy.io import fits
from astropy.table import Table, vstack, QTable
from astropy.time import Time
from astropy.units import Quantity
from gammapy.utils.scripts import make_path
from gammapy.utils.time import (
    time_ref_from_dict,
    time_ref_to_dict,
    time_relative_to_ref,
)

__all__ = ["GTI"]


class GTI:
    """Good time intervals (GTI) `~astropy.table.Table`.

    Data format specification: :ref:`gadf:iact-gti`

    Note: at the moment dead-time and live-time is in the
    EVENTS header ... the GTI header just deals with
    observation times.

    Parameters
    ----------
    table : `~astropy.table.Table`
        GTI table

    Examples
    --------
    Load GTIs for a H.E.S.S. event list:

    >>> from gammapy.data import GTI
    >>> gti = GTI.read('$GAMMAPY_DATA/hess-dl3-dr1//data/hess_dl3_dr1_obs_id_023523.fits.gz')
    >>> print(gti)
    GTI info:
    - Number of GTIs: 1
    - Duration: 1687.0 s
    - Start: 53343.92234009259 MET
    - Start: 2004-12-04T22:08:10.184 (time standard: TT)
    - Stop: 53343.94186555556 MET
    - Stop: 2004-12-04T22:36:17.184 (time standard: TT)

    Load GTIs for a Fermi-LAT event list:

    >>> gti = GTI.read("$GAMMAPY_DATA/fermi-3fhl-gc/fermi-3fhl-gc-events.fits.gz")
    >>> print(gti)
    GTI info:
    - Number of GTIs: 39042
    - Duration: 183139597.9032163 s
    - Start: 54682.65603794185 MET
    - Start: 2008-08-04T15:44:41.678 (time standard: TT)
    - Stop: 57236.96833546296 MET
    - Stop: 2015-08-02T23:14:24.184 (time standard: TT)
    """

    def __init__(self, table, time_ref=None):
        table = self._validate_table(table)
        self._table = table
        if time_ref is None:
            time_ref = self.table["TSTART"][0]
        self._time_ref = time_ref

    @property
    def table(self):
        """The table containing start and stop times."""
        return self._table

    @staticmethod
    def _validate_table(table):
        # Check that tstart is smaller than tstop?
        required_columns = ["TSTART", "TSTOP"]
        if not set(required_columns).issubset(table.colnames):
            raise KeyError(f"GTI table missing key words.")
        if not isinstance(table["TSTART"], Time) or not isinstance(table["TSTOP"], Time):
            raise TypeError(f"GTI table column do not contain Time objects.")
        # Set time format to MJD
        table["TSTART"].format="mjd"
        table["TSTOP"].format="mjd"
        return table

    def copy(self):
        return copy.deepcopy(self)

    @classmethod
    def create(cls, start, stop, reference_time="2000-01-01"):
        """Creates a GTI table from start and stop times.

        Parameters
        ----------
        start : `~astropy.units.Quantity`
            start times w.r.t. reference time
        stop : `~astropy.units.Quantity`
            stop times w.r.t. reference time
        reference_time : `~astropy.time.Time`
            the reference time to use in GTI definition
        """
        start = Quantity(start, ndmin=1)
        stop = Quantity(stop, ndmin=1)
        reference_time = Time(reference_time)
        table = Table({"TSTART": reference_time+start, "TSTOP": reference_time+stop})
        return cls(table, time_ref=reference_time)

    @classmethod
    def read(cls, filename, hdu="GTI"):
        """Read from FITS file.

        Parameters
        ----------
        filename : `pathlib.Path`, str
            Filename
        hdu : str
            hdu name. Default GTI.
        """
        filename = make_path(filename)
        input_table = Table.read(filename, hdu=hdu)
        time_ref = time_ref_from_dict(input_table.meta)
        start = time_ref + Quantity(input_table["START"].astype("float64"), "second")
        stop = time_ref + Quantity(input_table["STOP"].astype("float64"), "second")
        table = QTable(data={"TSTART": start, "TSTOP": stop})
        return cls(table, time_ref=time_ref)

    def write(self, filename, **kwargs):
        """Write to file.

        Parameters
        ----------
        filename : str or `Path`
            File name to write to.
        """
        hdu = fits.BinTableHDU(self.table, name="GTI")
        hdulist = fits.HDUList([fits.PrimaryHDU(), hdu])
        hdulist.writeto(make_path(filename), **kwargs)

    def __str__(self):
        return (
            "GTI info:\n"
            f"- Number of GTIs: {len(self.table)}\n"
            f"- Duration: {self.time_sum}\n"
            f"- Start: {self.time_start[0]} MET\n"
            f"- Start: {self.time_start[0].fits} (time standard: {self.time_start[-1].scale.upper()})\n"
            f"- Stop: {self.time_stop[-1]} MET\n"
            f"- Stop: {self.time_stop[-1].fits} (time standard: {self.time_stop[-1].scale.upper()})\n"
        )

    @property
    def time_delta(self):
        """GTI durations in seconds (`~astropy.units.Quantity`)."""
        delta = self.time_stop - self.time_start
        return delta.to('s')

    @property
    def time_ref(self):
        """Time reference (`~astropy.time.Time`)."""
        return self._time_ref
        # time_ref_from_dict(self.table.meta)

    @property
    def time_sum(self):
        """Sum of GTIs in seconds (`~astropy.units.Quantity`)."""
        return self.time_delta.sum()

    @property
    def time_start(self):
        """GTI start times (`~astropy.time.Time`)."""
        return self.table["TSTART"]

    @property
    def time_stop(self):
        """GTI end times (`~astropy.time.Time`)."""
        return self.table["TSTOP"]

    @property
    def start(self):
        """GTI start time delta since reference time in sec (`~astropy.Quantity`)."""
        return (self.time_start - self.time_ref).to('s')

    @property
    def stop(self):
        """GTI stop time delta since reference time in sec (`~astropy.Quantity`)."""
        return (self.time_stop - self.time_ref).to('s')

    @property
    def time_intervals(self):
        """List of time intervals"""
        return [
            (t_start, t_stop)
            for t_start, t_stop in zip(self.time_start, self.time_stop)
        ]

    @classmethod
    def from_time_intervals(cls, time_intervals, reference_time="2000-01-01"):
        """From list of time intervals

        Parameters
        ----------
        time_intervals : list of `~astropy.time.Time` objects
            Time intervals
        reference_time : `~astropy.time.Time`
            Reference time to use in GTI definition

        Returns
        -------
        gti : `GTI`
            GTI table.
        """
        reference_time = Time(reference_time)
        start = Time([_[0] for _ in time_intervals]) - reference_time
        stop = Time([_[1] for _ in time_intervals]) - reference_time
        meta = time_ref_to_dict(reference_time)
        table = Table({"START": start.to("s"), "STOP": stop.to("s")}, meta=meta)
        return cls(table=table)

    def select_time(self, time_interval):
        """Select and crop GTIs in time interval.

        Parameters
        ----------
        time_interval : `astropy.time.Time`
            Start and stop time for the selection.

        Returns
        -------
        gti : `GTI`
            Copy of the GTI table with selection applied.
        """
        # get GTIs that fall within the time_interval
        mask = self.time_start < time_interval[1]
        mask &= self.time_stop > time_interval[0]

        gti_crop = {}
        gti_crop["TSTART"] = Time(np.clip(self.time_start[mask], time_interval[0], time_interval[1]))
        gti_crop["TSTOP"] = Time(np.clip(self.time_stop[mask], time_interval[0], time_interval[1]))

        return self.__class__(Table(gti_crop), time_ref=self.time_ref)

    def stack(self, other):
        """Stack with another GTI in place.

        This simply stacks the two tables. No logic is applied to the intervals.
        The reference time of the first GTI is kept.

        Parameters
        ----------
        other : `~gammapy.data.GTI`
            GTI to stack to self

        """
        self._table = vstack([self.table, other.table])

    @classmethod
    def from_stack(cls, gtis, **kwargs):
        """Stack (concatenate) list of event lists.

        Calls `~astropy.table.vstack`.

        Parameters
        ----------
        gtis : list of `GTI`
            List of good time intervals to stack
        **kwargs : dict
            Keywords passed on to `~astropy.table.vstack`

        Returns
        -------
        gti : `GTI`
            Stacked good time intervals.
        """
        tables = [_.table for _ in gtis]
        stacked_table = vstack(tables, **kwargs)
        return cls(stacked_table)

    def union(self, overlap_ok=True, merge_equal=True):
        """Union of overlapping time intervals.

        Returns a new `~gammapy.data.GTI` object.

        Parameters
        ----------
        overlap_ok : bool
            Whether to raise an error when overlapping time bins are found.
        merge_equal : bool
            Whether to merge touching time bins e.g. ``(1, 2)`` and ``(2, 3)``
            will result in ``(1, 3)``.
        """
        # Algorithm to merge overlapping intervals is well-known,
        # see e.g. https://stackoverflow.com/a/43600953/498873

        table = self.table.copy()
        table.sort("TSTART")

        compare = lt if merge_equal else le

        # We use Python dict instead of astropy.table.Row objects,
        # because on some versions modifying Row entries doesn't behave as expected
        merged = [{"TSTART": table[0]["TSTART"], "TSTOP": table[0]["TSTOP"]}]
        for row in table[1:]:
            interval = {"TSTART": row["TSTART"], "TSTOP": row["TSTOP"]}
            if compare(merged[-1]["TSTOP"], interval["TSTART"]):
                merged.append(interval)
            else:
                if not overlap_ok:
                    raise ValueError("Overlapping time bins")

                merged[-1]["TSTOP"] = max(interval["TSTOP"], merged[-1]["TSTOP"])

        merged = Table(rows=merged, names=["TSTART", "TSTOP"], meta=self.table.meta)
        return self.__class__(merged)

    def group_table(self, time_intervals, atol="1e-6 s"):
        """Compute the table with the info on the group to which belong each time interval.

        The t_start and t_stop are stored in MJD from a scale in "utc".

        Parameters
        ----------
        time_intervals : list of `astropy.time.Time`
            Start and stop time for each interval to compute the LC
        atol : `~astropy.units.Quantity`
            Tolerance value for time comparison with different scale. Default 1e-6 sec.

        Returns
        -------
        group_table : `~astropy.table.Table`
            Contains the grouping info.
        """
        atol = Quantity(atol)

        group_table = Table(
            names=("group_idx", "time_min", "time_max", "bin_type"),
            dtype=("i8", "f8", "f8", "S10"),
        )
        time_intervals_lowedges = Time(
            [time_interval[0] for time_interval in time_intervals]
        )
        time_intervals_upedges = Time(
            [time_interval[1] for time_interval in time_intervals]
        )

        for t_start, t_stop in zip(self.time_start, self.time_stop):
            mask1 = t_start >= time_intervals_lowedges - atol
            mask2 = t_stop <= time_intervals_upedges + atol
            mask = mask1 & mask2
            if np.any(mask):
                group_index = np.where(mask)[0]
                bin_type = ""
            else:
                group_index = -1
                if np.any(mask1):
                    bin_type = "overflow"
                elif np.any(mask2):
                    bin_type = "underflow"
                else:
                    bin_type = "outflow"
            group_table.add_row(
                [group_index, t_start.utc.mjd, t_stop.utc.mjd, bin_type]
            )

        return group_table
