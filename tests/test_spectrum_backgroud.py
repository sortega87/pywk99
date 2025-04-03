import numpy as np
import xarray as xr

from pywk99.spectrum.background import get_background_spectrum


def test_background_spectrum_conserves_power():
    # construct test spectrum
    spectrum = xr.DataArray(0.0,
                            dims=["frequency", "wavenumber"],
                            coords={"frequency": np.arange(-10, 10)/10.0,
                                    "wavenumber": np.arange(15, -15, -1)})
    spectrum[10:15, 5:20] = 1
    # test
    test_background_spectrum = get_background_spectrum(spectrum, spectrum)
    assert test_background_spectrum.sum() == spectrum.sum()