#!/usr/bin/env python3

import numpy as np

from matplotlib.dates import date2num
from scipy.interpolate import interpn


def density_scatter(a, times, values, bins=(1000, 200), **kwargs):
    x = date2num(times)
    y = np.log(values)
    finite = np.isfinite(y)
    counts, x_edges, y_edges = np.histogram2d(
        x[finite], y[finite], bins=bins, density=False
    )
    area = np.diff(x_edges).reshape((-1, 1)) * np.diff(y_edges).reshape((1, -1))
    density = counts / counts.sum(axis=1).reshape((-1, 1)) / area
    z = interpn(
        (0.5 * (x_edges[1:] + x_edges[:-1]), 0.5 * (y_edges[1:] + y_edges[:-1])),
        density,
        np.vstack([x, y]).T,
        method="nearest",
        bounds_error=False,
    )

    z[np.where(np.isnan(z))] = 0
    i = z.argsort()
    times, values, z = times[i], values[i], z[i]
    a.scatter(times, values, c=z, **kwargs)
