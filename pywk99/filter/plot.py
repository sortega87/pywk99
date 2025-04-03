"""Plot filter information on Spectrum figures."""

from typing import Optional, Union
from matplotlib import pyplot as plt
from shapely.geometry import Polygon, MultiPolygon

from pywk99.filter.window import FilterWindow

def plot_filter_window(filter_window: FilterWindow,
                       ax: Optional[plt.Axes] = None,
                       fill: bool = False,
                       **plot_kwargs,
                       ):
    """Plot the wave filter window."""
    if ax is None:
        ax = plt.gca()
    PLOT_FUNCTIONS = {False: ax.plot, True: ax.fill}
    plot_function = PLOT_FUNCTIONS[fill]
    if not plot_kwargs:
        plot_kwargs = {"color": 'dodgerblue'}
    if not "color" in plot_kwargs.keys():
        plot_kwargs["color"] = 'dodgerblue'
    if isinstance(filter_window.polygon, Polygon):
        x, y = filter_window.polygon.exterior.xy
        plot_function(x, y, **plot_kwargs)
    if isinstance(filter_window.polygon, MultiPolygon):
        for geom in filter_window.polygon.geoms:
            x, y = geom.exterior.xy
            plot_function(x, y, **plot_kwargs)