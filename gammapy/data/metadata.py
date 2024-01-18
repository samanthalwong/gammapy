# Licensed under a 3-clause BSD style license - see LICENSE.rst
from typing import ClassVar, Literal, Optional
from pydantic import Field
from gammapy.utils.fits import earth_location_from_dict
from gammapy.utils.metadata import (
    METADATA_FITS_KEYS,
    CreatorMetaData,
    MetaData,
    ObsInfoMetaData,
    PointingInfoMetaData,
    TargetMetaData,
)
from gammapy.utils.types import EarthLocationType, TimeType

__all__ = ["ObservationMetaData"]

OBSERVATION_METADATA_FITS_KEYS = {
    "location": {
        "input": lambda v: earth_location_from_dict(v),
        "output": lambda v: {
            "GEOLON": v.lon.deg,
            "GEOLAT": v.lat.deg,
            "ALTITUDE": v.height.to_value("m"),
        },
    },
    "deadtime_fraction": {
        "input": lambda v: 1 - v["DEADC"],
        "output": lambda v: {"DEADC": 1 - v},
    },
}

METADATA_FITS_KEYS["observation"] = OBSERVATION_METADATA_FITS_KEYS


class ObservationMetaData(MetaData):
    """Metadata containing information about the Observation.

    Parameters
    ----------
    obs_info : `~gammapy.utils.ObsInfoMetaData`
        The general observation information.
    pointing : `~gammapy.utils.PointingInfoMetaData
        The pointing metadata.
    target : `~gammapy.utils.TargetMetaData
        The target metadata.
    creation : `~gammapy.utils.CreatorMetaData`
        The creation metadata.
    location : `~astropy.coordinates.EarthLocation` or str, optional
        The observatory location.
    deadtime_fraction : float
        The observation deadtime fraction. Default is 0.
    time_start : `~astropy.time.Time` or str
        The observation start time.
    time_stop : `~astropy.time.Time` or str
        The observation stop time.
    reference_time : `~astropy.time.Time` or str
        The observation reference time.
    optional : dict, optional
        Additional optional metadata.
    """

    _tag: ClassVar[Literal["observation"]] = "observation"
    obs_info: Optional[ObsInfoMetaData] = None
    pointing: Optional[PointingInfoMetaData] = None
    target: Optional[TargetMetaData] = None
    location: Optional[EarthLocationType] = None
    deadtime_fraction: float = Field(0.0, ge=0, le=1.0)
    time_start: Optional[TimeType] = None
    time_stop: Optional[TimeType] = None
    reference_time: Optional[TimeType] = None
    creation: Optional[CreatorMetaData] = None
    optional: Optional[dict] = None

    @classmethod
    def from_header(cls, header, format="gadf"):
        meta = super(ObservationMetaData, cls).from_header(header, format)

        meta.creation = CreatorMetaData()
        # Include additional gadf keywords not specified as ObservationMetaData attributes
        optional_keywords = [
            "OBSERVER",
            "EV_CLASS",
            "TELAPSE",
            "TELLIST",
            "N_TELS",
            "TASSIGN",
            "DST_VER",
            "ANA_VER",
            "CAL_VER",
            "CONV_DEP",
            "CONV_RA",
            "CONV_DEC",
            "TRGRATE",
            "ZTRGRATE",
            "MUONEFF",
            "BROKPIX",
            "AIRTEMP",
            "PRESSURE",
            "RELHUM",
            "NSBLEVEL",
        ]
        optional = dict()
        for key in optional_keywords:
            if key in header.keys():
                optional[key] = header[key]
        meta.optional = optional

        return meta
