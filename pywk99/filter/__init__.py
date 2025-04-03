"""Filter a signal for a given wavenumber-frequency band"""
from pywk99.filter.filter import filter_variable
from pywk99.filter.window import FilterWindow
from pywk99.filter.window import get_wave_filter_window
from pywk99.filter.window import get_box_filter_window
from pywk99.filter.window import get_tropical_depression_window
from pywk99.filter.window import get_mjo_window
from pywk99.filter.plot import plot_filter_window
