"""Take Discrete Fourier Transforms of Variables."""

from typing import Any
import xarray as xr
import numpy as np


def fourier_transform(variable: xr.DataArray) -> xr.DataArray:
    """Take the Discrete Fourier Transform of a variable."""
    variable = variable.transpose("time", "lon", ...)
    variable_array = variable.values
    spectrum_array = np.fft.fftn(variable_array, axes=[0, 1])
    spectrum_array = np.fft.fftshift(spectrum_array, axes=[0, 1])
    # construct dataarray
    frequency = _get_frequencies(variable)
    wavenumbers = _get_wavenumbers(variable)
    remaining_dims = list(variable.dims)[2:]
    new_dims = ["frequency", "wavenumber"] + remaining_dims
    new_coords = ([("frequency", frequency, {"units": "Cycles per Day"}),
                   ("wavenumber", wavenumbers, {"units": "Zonal Wavenumber"})] +
                  [(dim, variable[dim].data, variable[dim].attrs)
                   for dim in remaining_dims])
    spectrum = xr.DataArray(
        data=spectrum_array,
        dims=new_dims,
        coords=new_coords
    )
    return spectrum


def inverse_fourier_transform(spectrum: xr.DataArray,
                              xarray_coords) -> xr.DataArray:
    """Take the Inverse Discrete Fourier Transform of a spectrum."""
    if not 'lon' in xarray_coords and not 'time' in xarray_coords:
        raise ValueError("Either dimension 'lon' or dimension 'time' not " +
                         "present in coords!")
    spectrum_array = spectrum.transpose("frequency", "wavenumber", ...).values
    variable_array = np.fft.ifftshift(spectrum_array, axes=[0, 1])
    variable_array = np.real(np.fft.ifftn(variable_array, axes=[0, 1]))
    # construct dataarray
    remaining_dims = list(spectrum.dims)[2:]
    new_dims = ["time", "lon"] + remaining_dims
    new_coords = ([xarray_coords['time'], xarray_coords['lon']] +
                  [(dim, spectrum[dim].data, spectrum[dim].attrs)
                   for dim in remaining_dims])
    variable = xr.DataArray(data=variable_array,
                            coords=new_coords,
                            dims=new_dims
                            )
    variable = variable.transpose("time", "lon", ...)
    return variable


def _get_frequencies(variable: xr.DataArray) -> np.ndarray:
    """Get the frequency in cycles per day after a numpy fftshift."""
    period_timedelta = variable.time[-1].values - variable.time[0].values
    period = period_timedelta.astype("timedelta64[D]").astype(int)
    n_time = len(variable.time)
    frequency = _fourier_sequence(n_time)/period
    return frequency


def _get_wavenumbers(variable: xr.DataArray) -> np.ndarray:
    """Get the zonal wavenumbers of the fft spectrum after a numpy fftshift."""
    n_lon = len(variable.lon)
    wavenumbers = _fourier_sequence(n_lon)
    wavenumbers = - wavenumbers  # positive wave number is eastward in WK99
    return wavenumbers


def _fourier_sequence(n_len: int) -> np.ndarray:
    """Get the DFT number sequence after a numpy fftshift."""
    fourier_sequence = n_len*np.fft.fftfreq(n_len)
    fourier_sequence = np.fft.fftshift(fourier_sequence)
    fourier_sequence = np.round(fourier_sequence).astype(int)
    return fourier_sequence
