"""
Test waves dispersion relations against those reported in literature.

Waves are tested against Figure 1 of Kiladis, George N., et al. "Convectively
coupled equatorial waves." Reviews of Geophysics 47.2 (2009).
"""


import pytest
from pywk99.waves.waves import LinearWave
import numpy as np


SYMMETRIC_WAVE_TYPES = ["inertio_gravity", "kelvin", "equatorial_rossby"]
ASYMMETRIC_WAVE_TYPES = ["inertio_gravity", "mixed_rossby_gravity"]


@pytest.mark.parametrize(
    "wave_type, equivalent_depth, n_polynomial, test_wavenumber, omega_expected",
    [("kelvin", 25, 1, 0.0, 0.0),
     ("kelvin", 25, 1, 10, 0.33),
     ("inertio_gravity", 25, 1, 0.0, 0.45),
     ("inertio_gravity", 25, 1, 10, 0.59),
     ("inertio_gravity", 25, 1, -10, 0.525),
     ("equatorial_rossby", 25, 1, 0.0, 0.0),
     ("equatorial_rossby", 25, 1, -10, 0.075),
     ("equatorial_rossby", 25, 1, -20, 0.075),
     ("inertio_gravity", 25, 2, 0.0, 0.575),
     ("inertio_gravity", 25, 2, 10, 0.7),
     ("inertio_gravity", 25, 2, -10, 0.65),
     ("mixed_rossby_gravity", 25, 0, -20, 0.08),
     ("mixed_rossby_gravity", 25, 0, 20, 0.76),
     ("gravity", 25, 1, -20, 0.68),
     ("gravity", 25, 1, 20, 0.68)])
def test_waves_dispersion_relation(wave_type, equivalent_depth, n_polynomial,
                                   test_wavenumber, omega_expected):
    linearwave = LinearWave(wave_type, equivalent_depth, n_polynomial)
    wavenumber = np.array([test_wavenumber])
    omega = linearwave.frequency(wavenumber)[0]
    assert omega == pytest.approx(omega_expected, abs=0.05)


@pytest.mark.parametrize("wave_type, test_wavenumber",
                         [("kelvin", -10.0),
                          ("equatorial_rossby", 10)])
def test_not_defined_dispersion_relation_return_nan(wave_type,
                                                    test_wavenumber):
    linearwave = LinearWave(wave_type, equivalent_depth=25, n_polynomial=1)
    wavenumber = np.array([test_wavenumber])
    omega = linearwave.frequency(wavenumber)[0]
    assert np.isnan(omega)
