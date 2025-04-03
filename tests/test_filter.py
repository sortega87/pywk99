import pandas as pd
import pytest

import numpy as np
import xarray as xr

from pywk99.filter.filter import filter_variable
from pywk99.filter.filter import modify_spectrum
from pywk99.filter.window import get_box_filter_window


@pytest.fixture
def variable():
    """Variable with frequency 15/360 CPD and wavenumber 5"""
    time = pd.date_range("2020-01-01", freq="1D", periods=365)
    lat = np.arange(-15, 15)
    lon = np.arange(360)
    data = np.ones(shape=(len(time), len(lat), len(lon)))
    time_cycle = np.cos(15*2*np.pi*np.arange(len(time))/len(time))
    lon_cycle = np.cos(5*2*np.pi*np.arange(len(lon))/len(lon))
    data = 10 + ((data*lon_cycle).T*time_cycle).T
    variable = xr.DataArray(data,
                            dims=["time", "lat", "lon"],
                            coords={"time": time,
                                    "lat": lat,
                                    "lon": lon})
    variable = variable.transpose("time", "lon", "lat")
    return variable


def test_filter_removes_unwanted_window(variable):
    filter_window = get_box_filter_window(0, 10, 0/360, 5/360)
    test_filtered_variable = filter_variable(variable, filter_window)
    assert (2*test_filtered_variable).std() < 0.1/100


def test_filter_keeps_wanted_window(variable):
    filter_window = get_box_filter_window(-10, 10, 10/360, 20/360)
    test_filtered_variable = filter_variable(variable, filter_window,
                                             taper=False)
    assert ((variable-10) - test_filtered_variable).std() < 0.1/100


def test_filter_multiple_removes_unwanted_window(variable):
    filter_windows = [get_box_filter_window(0, 10, 0/360, 5/360),
                      get_box_filter_window(0, 10, 0/360, 5/360)]
    test_filtered_variable = filter_variable(variable,
                                             filter_windows=filter_windows)
    assert (2*test_filtered_variable.box).std() < 0.1/100
    assert (2*test_filtered_variable.box2).std() < 0.1/100


def test_filter_multiple_keeps_wanted_window(variable):
    filter_windows = [get_box_filter_window(-10, 10, 10/360, 20/360),
                      get_box_filter_window(-10, 10, 10/360, 20/360)]
    test_filtered_variable = filter_variable(variable,
                                             filter_windows=filter_windows,
                                             taper=False)
    assert ((variable-10) - test_filtered_variable.box).std() < 0.1/100
    assert ((variable-10) - test_filtered_variable.box2).std() < 0.1/100


def test_modify_spectrum_filtering():
    # construct test modified spectrum
    spectrum = xr.DataArray(1.0,
                            dims=["frequency", "wavenumber"],
                            coords={"frequency": np.arange(-10, 10)/10.0,
                                    "wavenumber": np.arange(15, -15, -1)})
    filter_window = get_box_filter_window(1, 10, 0.1, 0.5)
    modified_spectrum = modify_spectrum(spectrum, filter_window, 'filter')
    # construct expected spectrum
    expected_spectrum = xr.zeros_like(spectrum)
    wavenumber_index_1 = np.logical_and(spectrum.wavenumber.values >= 1,
                                        spectrum.wavenumber.values <= 10)
    wavenumber_index_2 = np.logical_and(spectrum.wavenumber.values >= -10,
                                        spectrum.wavenumber.values <= -1)
    frequency_index_1 = np.logical_and(spectrum.frequency.values >= 0.1,
                                    spectrum.frequency.values <= 0.5)
    frequency_index_2 = np.logical_and(spectrum.frequency.values >= -0.5,
                                    spectrum.frequency.values <= -0.1)
    expected_spectrum[frequency_index_1, wavenumber_index_1] = 1.0
    expected_spectrum[frequency_index_2, wavenumber_index_2] = 1.0
    # test
    assert np.all(np.all(expected_spectrum == modified_spectrum))


def test_modify_spectrum_substract():
    # construct test modified spectrum
    spectrum = xr.DataArray(1.0,
                            dims=["frequency", "wavenumber"],
                            coords={"frequency": np.linspace(0, 1, 11),
                                    "wavenumber": np.arange(15, -15, -1)})
    filter_window = get_box_filter_window(1, 10, 0.1, 0.5)
    modified_spectrum = modify_spectrum(spectrum, filter_window, 'substract')
    # expected filtered regions
    wavenumber_index = np.logical_and(spectrum.wavenumber.values >= 1,
                                        spectrum.wavenumber.values <= 10)
    frequency_index = np.logical_and(spectrum.frequency.values >= 0.1,
                                     spectrum.frequency.values <= 0.5)
    # test
    assert np.all(modified_spectrum[frequency_index, wavenumber_index] == 0.0)
    assert np.all(modified_spectrum[~frequency_index, ~wavenumber_index] == 1.0)


def test_filter_runs_without_removing_trends_or_seasons(variable):
    filter_window = get_box_filter_window(0, 10, 0/360, 5/360)
    test_filtered_variable = filter_variable(variable, filter_window,
                                             rm_seasonal_cycle=False,
                                             rm_linear_trend=False)
    assert np.any(test_filtered_variable)


def test_multiple_filter_runs_without_removing_trends_or_seasons(variable):
    filter_windows = [get_box_filter_window(-10, 10, 10/360, 20/360),
                      get_box_filter_window(-10, 10, 10/360, 20/360)]
    test_filtered_variable = filter_variable(variable,
                                             filter_windows=filter_windows,
                                             taper=False,
                                             rm_seasonal_cycle=False,
                                             rm_linear_trend=False)
    assert np.any(test_filtered_variable)