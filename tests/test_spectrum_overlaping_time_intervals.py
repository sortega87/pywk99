import pytest

import numpy as np

from pywk99.spectrum._spectrum import _get_successive_overlapping_time_intervals


@pytest.fixture
def one_year_intervals():
    interval_times = _get_successive_overlapping_time_intervals(
        start=np.datetime64("2000-01-01 00:00:00"),
        end=np.datetime64("2001-01-01 00:00:00"),
        window_length=np.timedelta64(96, "D"),
        overlap_length=np.timedelta64(60, "D"),
        data_frequency=np.timedelta64(3, "h")
    )
    return interval_times


def test_interval_length(one_year_intervals):
    expected_interval_length = np.timedelta64(96, "D") - np.timedelta64(3, "h")
    time_deltas = [
        (interval[1] - interval[0]) == expected_interval_length
        for interval in one_year_intervals
    ]
    assert all(time_deltas)


def test_interval_overlap(one_year_intervals):
    expected_interval_overlap = (
        np.timedelta64(60, "D") - np.timedelta64(3, "h")
        )
    overlap_deltas = [
        (first_interval[1] - second_interval[0]) == expected_interval_overlap
        for first_interval, second_interval in zip(
            one_year_intervals[:-1], one_year_intervals[1:]
        )
    ]
    assert all(overlap_deltas)


def test_interval_start(one_year_intervals):
    assert one_year_intervals[0][0] == np.datetime64("2000-01-01 00:00:00")


def test_interval_end(one_year_intervals):
    assert one_year_intervals[-1][1] <= np.datetime64("2001-01-01 00:00:00")


def test_number_of_intervals(one_year_intervals):
    assert len(one_year_intervals) == 8


def test_start_after_end_raises_error():
    with pytest.raises(ValueError):
        _get_successive_overlapping_time_intervals(
            start=np.datetime64("2001-01-01 00:00:00"),
            end=np.datetime64("2000-01-01 00:00:00"),
            window_length=np.timedelta64(1, "h"),
            overlap_length=np.timedelta64(1, "h"),
            data_frequency=np.timedelta64(3, "h")
        )


def test_larger_overlap_length_raises_error():
    with pytest.raises(ValueError):
        _get_successive_overlapping_time_intervals(
            start=np.datetime64("2000-01-01 00:00:00"),
            end=np.datetime64("2000-01-02 00:00:00"),
            window_length=np.timedelta64(1, "h"),
            overlap_length=np.timedelta64(5, "h"),
            data_frequency=np.timedelta64(3, "h")
        )


def test_small_data_length_relative_to_window_raises_error():
    with pytest.raises(ValueError):
        _get_successive_overlapping_time_intervals(
            start=np.datetime64("2000-01-01 00:00:00"),
            end=np.datetime64("2000-01-02 00:00:00"),
            window_length=np.timedelta64(3, "D"),
            overlap_length=np.timedelta64(5, "h"),
            data_frequency=np.timedelta64(3, "h")
        )


def test_overlapping_segments_for_exact_interval_number():
    interval_times = _get_successive_overlapping_time_intervals(
        start=np.datetime64("2000-01-01 00:00:00"),
        end=np.datetime64("2000-01-01 10:00:00"),
        window_length=np.timedelta64(6, "h"),
        overlap_length=np.timedelta64(4, "h"),
        data_frequency=np.timedelta64(3, "h")
    )
    expected_intervals = [
        (np.datetime64("2000-01-01 00:00:00"),
         np.datetime64("2000-01-01 06:00:00") - np.timedelta64(3, "h")),
        (np.datetime64("2000-01-01 02:00:00"),
         np.datetime64("2000-01-01 08:00:00") - np.timedelta64(3, "h")),
        (np.datetime64("2000-01-01 04:00:00"),
         np.datetime64("2000-01-01 10:00:00") - np.timedelta64(3, "h")),
        (np.datetime64("2000-01-01 06:00:00"),
         np.datetime64("2000-01-01 12:00:00") - np.timedelta64(3, "h")),
    ]
    assert interval_times == expected_intervals


def test_overlapping_segments_for_a_month():
    interval_times = _get_successive_overlapping_time_intervals(
        start=np.datetime64("2004-04-01 00:00:00"),
        end=np.datetime64("2004-04-30 21:00:00"),
        window_length=np.timedelta64(30, "D"),
        overlap_length=np.timedelta64(0, "D"),
        data_frequency=np.timedelta64(3, "h")
    )
    expected_intervals = [(np.datetime64('2004-04-01T00:00:00.000000000'),
                          np.datetime64('2004-04-30T21:00:00.000000000'))]
    assert interval_times == expected_intervals