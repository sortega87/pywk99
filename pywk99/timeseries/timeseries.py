"""Remove the mean and seasonal cycles of time series."""

from typing import Union
import numpy as np
import xarray as xr


def remove_linear_trend(
        variable: Union[xr.DataArray, xr.Dataset],
        keep_mean: bool = False) -> Union[xr.DataArray, xr.Dataset]:
    """Remove the mean and linear trend along the time dimension."""
    new_variable = variable.copy()
    polyfit_results = new_variable.polyfit(dim="time", deg=1)
    if isinstance(variable, xr.DataArray):
        new_variable = _remove_linear_trend_datarray(new_variable,
                                                     polyfit_results)
    elif isinstance(variable, xr.Dataset):
        new_variable = _remove_linear_trend_dataset(new_variable,
                                                    polyfit_results)
    else:
        raise ValueError(f"Unknown variable type: '{variable}'")
    if keep_mean:
        new_variable = new_variable + variable.mean(dim='time')
    return new_variable


def remove_seasonal_cycle(
        variable: Union[xr.DataArray, xr.Dataset],
        keep_slow_var: bool = False,
        n_harmonics: int = 3) -> Union[xr.DataArray, xr.Dataset]:
    """Remove the seasonal cycle and normalize."""
    granularity = variable.time[1] - variable.time[0]
    data_time_span = variable.sizes["time"] * granularity
    data_time_span = data_time_span.astype("timedelta64[D]")
    k365 = int(data_time_span / np.timedelta64(365, 'D'))
    new_variable = variable.transpose("time", ...).copy()
    if isinstance(new_variable, xr.DataArray):
        new_variable = _remove_seasonal_cycle_dataarray(
            new_variable, keep_slow_var, n_harmonics, k365)
    elif isinstance(new_variable, xr.Dataset):
        new_variable = _remove_seasonal_cycle_dataset(new_variable,
                                                      keep_slow_var,
                                                      n_harmonics, k365)
    else:
        raise ValueError(f"Unknown variable type: '{variable}'")
    return new_variable


def taper_variable_time_ends(
        variable: Union[xr.DataArray, xr.Dataset],
        alpha: float = 0.5) -> Union[xr.DataArray, xr.Dataset]:
    """Taper the datarray data with a Tukey window."""
    length_array = len(variable.time)
    window_data = _get_tukey_window(length_array, alpha)
    hann_window_array = xr.DataArray(data=window_data,
                                     coords={"time": variable.time})
    new_variable = variable * hann_window_array
    return new_variable


def check_time_index_has_expected_frequency(
        variable: Union[xr.DataArray,
                        xr.Dataset], data_frequency: np.timedelta64) -> bool:
    """Raise ValueError if data_frequency is not the same for all times"""
    time_deltas = variable.time.to_series().diff()[1:]
    all_equal = (time_deltas == data_frequency).all()
    if not all_equal:
        raise ValueError("Incorrect data_frequency detected. Expecting "
                         f"{data_frequency.astype('timedelta64[h]')}.")
    return all_equal


def check_variable_coordinates_are_sorted(
        variable: Union[xr.DataArray, xr.Dataset]) -> None:
    """Raises ValueError if coordinates are not order in ascending order."""
    for dim in variable.dims:
        sorted = np.all(
            variable[dim].sortby(dim).values == variable[dim].values)
        if not sorted:
            raise ValueError(f"Dimension '{dim}' is not sorted")


def check_for_one_max_two_variables(
        variable: Union[xr.DataArray, xr.Dataset]) -> None:
    """Check for a maximum of two variables in the dataset"""
    if isinstance(variable, xr.DataArray):
        return
    if len(variable.data_vars) > 2:
        raise ValueError("'variable' has more than two data fields.")


def check_for_exactly_two_variables(
        variable: Union[xr.DataArray, xr.Dataset]) -> None:
    """Check for a maximum of two variables in the dataset"""
    if len(variable.data_vars) != 2:
        raise ValueError(("Exactly two variables must be provided. "
                          f"You provided '{len(variable.data_vars)}'"))


def convert_to_dataset(
        variable: Union[xr.DataArray, xr.Dataset]) -> xr.Dataset:
    """Converts a Datarray to a dataset for consistency within the library."""
    if isinstance(variable, xr.Dataset):
        return variable.copy()
    if not variable.name:
        variable.name = "variable"
    return xr.Dataset({variable.name: variable})


def _get_tukey_window(length: int, alpha: float) -> np.array:
    """Get tukey tapper window data."""
    if alpha > 1:
        raise ValueError("alpha must be in the interval [0, 1]")
    index = np.arange(length)
    window_data = np.ones(length)
    lobe_width = alpha * length / 2
    if alpha == 0:
        return window_data
    if lobe_width <= 1:
        raise ValueError("Insufficient window lobe width. Increase alpha.")
    start_index = np.where(index < lobe_width)[0]
    start_window_lenght = len(start_index)
    start_window_values = 0.5 * (1 - np.cos(2 * np.pi * start_index /
                                            (alpha * length)))
    window_data[start_index] = start_window_values
    window_data[-(start_window_lenght - 1):] = np.flip(start_window_values[1:])
    return window_data


def _remove_linear_trend_dataset(variable, polyfit_results):
    new_variable = xr.merge([(
        variable[var] -
        xr.polyval(variable["time"],
                   polyfit_results[f"{var}_polyfit_coefficients"])).rename(var)
                             for var in list(variable.keys())])

    return new_variable


def _remove_linear_trend_datarray(variable, polyfit_results):
    linear_fit = xr.polyval(variable["time"],
                            polyfit_results.polyfit_coefficients)
    new_variable = variable - linear_fit
    return new_variable


def _remove_seasonal_cycle_dataset(variable, keep_slow_var, n_harmonics, k365):
    for variable_name in list(variable.keys()):
        variable_datarray = variable[variable_name]
        variable_datarray = _remove_seasonal_cycle_dataarray(
            variable_datarray, keep_slow_var, n_harmonics, k365
            )
        variable[variable_name] = variable_datarray
    return variable


def _remove_seasonal_cycle_dataarray(variable, keep_slow_var, n_harmonics,
                                     k365):
    array = variable.values
    # select and remove the harmonics using the real fft
    fft_array = np.fft.rfft(array, axis=0)
    start = k365 if keep_slow_var else 0
    fft_array[start:k365 + n_harmonics] = 0.0
    filtered_array = np.fft.irfft(fft_array, n=len(array), axis=0)
    variable.values = filtered_array
    return variable
