"""Test timeseries functions."""
import pytest
import numpy as np
import pandas as pd
import xarray as xr

from pywk99.timeseries.timeseries import check_for_exactly_two_variables
from pywk99.timeseries.timeseries import check_for_one_max_two_variables
from pywk99.timeseries.timeseries import check_variable_coordinates_are_sorted
from pywk99.timeseries.timeseries import remove_linear_trend
from pywk99.timeseries.timeseries import remove_seasonal_cycle
from pywk99.timeseries.timeseries import taper_variable_time_ends


def test_remove_seasonal_cycle():
    # create test variable with 3 seasonal 1 high freq cycles
    time = pd.date_range("2020-01-01", "2021-12-31", freq="1D")
    time_index = np.arange(len(time))
    time_len = len(time)
    cycle0 = 100
    cycle1 = 100*np.cos(2*2*np.pi*time_index/time_len)
    cycle2 = 50*np.sin(3*2*np.pi*time_index/time_len)
    cycle3 = 100*np.cos(4*2*np.pi*time_index/time_len)
    cycle30 = 10*np.cos(13*2*np.pi*time_index/time_len)
    data = cycle0 + cycle1 + cycle2 + cycle3 + cycle30
    variable = xr.DataArray(data, dims=["time"], coords={"time": time})
    # test
    test_noseasonal_variable = remove_seasonal_cycle(variable)
    assert np.abs(cycle30 - test_noseasonal_variable).max() < 1E-12


def test_remove_seasonal_cycle_for_a_dataset():
    # create test variable with 3 seasonal 1 high freq cycles
    time = pd.date_range("2020-01-01", "2021-12-31", freq="1D")
    time_index = np.arange(len(time))
    time_len = len(time)
    cycle0 = 100
    cycle1 = 100*np.cos(2*2*np.pi*time_index/time_len)
    cycle2 = 50*np.sin(3*2*np.pi*time_index/time_len)
    cycle3 = 100*np.cos(4*2*np.pi*time_index/time_len)
    cycle30 = 10*np.cos(13*2*np.pi*time_index/time_len)
    data = cycle0 + cycle1 + cycle2 + cycle3 + cycle30
    variable = xr.DataArray(data, dims=["time"], coords={"time": time})
    variables = xr.Dataset({"var1": variable,
                            "var2": variable.copy()})
    # test
    test_non_seasonal_variables = remove_seasonal_cycle(variables)
    assert np.abs(cycle30 - test_non_seasonal_variables.var1).max() < 1E-12
    assert np.abs(cycle30 - test_non_seasonal_variables.var2).max() < 1E-12


def test_remove_linear_trend():
    # create test variable with trend and 1 cycle
    time = pd.date_range("2020-01-01", "2020-12-31", freq="1D")
    time_index = np.arange(len(time))
    time_len = len(time)
    mean = 100.0
    linear = time_index
    cycle = np.cos(5*2*np.pi*time_index/time_len)
    data = mean + linear + cycle
    variable = xr.DataArray(data, dims=["time"], coords={"time": time})
    # test
    test_timeseries_variable = remove_linear_trend(variable)
    assert np.abs(cycle - test_timeseries_variable).max() < 0.01


def test_remove_linear_trend_for_a_dataset():
    # create test variable with trend and 1 cycle
    time = pd.date_range("2020-01-01", "2020-12-31", freq="1D")
    time_index = np.arange(len(time))
    time_len = len(time)
    mean = 100.0
    linear = time_index
    cycle = np.cos(5*2*np.pi*time_index/time_len)
    data = mean + linear + cycle
    variable = xr.DataArray(data, dims=["time"], coords={"time": time})
    variables = xr.Dataset({"var1": variable,
                            "var2": variable.copy()})
    # test
    test_timeseries_variables = remove_linear_trend(variables)
    assert np.abs(cycle - test_timeseries_variables.var1).max() < 0.01
    assert np.abs(cycle - test_timeseries_variables.var2).max() < 0.01

def test_tukey_tappering():
    lenght = 8
    variable_segment = xr.DataArray(1.0,
                                    dims=["time"],
                                    coords={"time": range(lenght)})
    tapered_variable_segment = taper_variable_time_ends(variable_segment,
                                                        alpha=0.5)
    assert np.isclose(tapered_variable_segment[0], 0.0, 1E-12)
    assert np.isclose(tapered_variable_segment[1], 0.5, 1E-12)
    assert np.all(np.isclose(tapered_variable_segment[2:7], 1.0, 1E-12))
    assert np.isclose(tapered_variable_segment[7], 0.5, 1E-12)


def test_tukey_tappering_with_insuficient_alpha():
    lenght = 8
    variable_segment = xr.DataArray(1.0,
                                    dims=["time"],
                                    coords={"time": range(lenght)})
    with pytest.raises(ValueError):
        taper_variable_time_ends(variable_segment, alpha=0.001)


def test_unsorted_variable_check_raises_error():
    lon = [1, 0, -1, -2]
    lat = [0, 1, 2, 3]
    variable = xr.DataArray(10.0, dims=["lon", "lat"],
                            coords={"lon": lon, "lat": lat})
    print(variable)
    with pytest.raises(ValueError):
        check_variable_coordinates_are_sorted(variable)


def test_datarrays_pass_max_variable_test():
    variable = xr.DataArray(name="olr",
                            data=range(10),
                            dims=["time"],
                            coords={"time": range(10)})
    check_for_one_max_two_variables(variable)


def test_dataset_with_two_variables_passes_max_variable_test():
    variable = xr.DataArray(name="olr",
                            data=range(10),
                            dims=["time"],
                            coords={"time": range(10)}).to_dataset()
    variable["temp"] = variable.olr/10
    check_for_one_max_two_variables(variable)


def test_dataset_three_variables_raises_max_variable_error():
    with pytest.raises(ValueError):
        variable = xr.DataArray(name="olr",
                                data=range(10),
                                dims=["time"],
                                coords={"time": range(10)}).to_dataset()
        variable["temp"] = variable.olr/10
        variable["dwtemp"] = variable.olr/10
        check_for_one_max_two_variables(variable)


def test_one_variable_dataset_raises_error():
    with pytest.raises(ValueError):
        variable = xr.DataArray(name="olr",
                                data=range(10),
                                dims=["time"],
                                coords={"time": range(10)}).to_dataset()
        check_for_exactly_two_variables(variable)
