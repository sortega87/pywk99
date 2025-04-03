import xarray as xr
import numpy as np

def sym_lat_taper(
        variable: xr.DataArray,
        zero_crossing: float,
        ) -> xr.DataArray:
    """
    Tapers the data with a cosine function in latitude, thereby highlighting the
    symmetric component of a field (symmetric with respect to the equator).

    Inputs:
    -------
    variable: xr.DataArray
        Data which should be tapered
    
    zero_crossing: float
        Latitude at which the data should be tapered to zero

    Output:
    -------
    tapered_variable: xr.DataArray
        Tapered data
    """
    period_factor = 90./zero_crossing
    smooth = np.cos(period_factor * np.deg2rad(variable.lat))
    return smooth * variable


def asym_lat_taper(
        variable: xr.DataArray,
        zero_crossing: float,
        ) -> xr.DataArray:
    """
    Tapers the data with a cosine function in latitude, thereby highlighting the
    symmetric component of a field (symmetric with respect to the equator).

    Inputs:
    -------
    variable: xr.DataArray
        Data which should be tapered
    
    zero_crossing: float
        Latitude at which the data should be tapered to zero

    Output:
    -------
    tapered_variable: xr.DataArray
        Tapered data
    """
    period_factor = 90./zero_crossing
    smooth = np.abs(np.sin(period_factor * np.deg2rad(variable.lat)))
    return smooth * variable