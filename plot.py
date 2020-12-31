#!/usr/bin/env python3

import argparse
import matplotlib as mpl
import matplotlib.pyplot as pp
import numpy as np
import sys

from curl import read_curl
from datetime import datetime, timedelta
from matplotlib.dates import date2num
from pathlib import Path
from ping import read_latencies
from scipy.interpolate import interpn, interp1d


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


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("ping", type=Path)
    parser.add_argument("curl", type=Path)
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        metavar="SECONDS",
        help="How many seconds of history to process",
    )
    args = parser.parse_args(argv[1:])

    if args.time:
        target_time = datetime.now() - timedelta(seconds=args.time)
    else:
        target_time = datetime.fromtimestamp(0)

    durations = read_curl(args.curl, target_time)
    durations_isfinite = np.isfinite(durations)
    durations_finite = durations[durations_isfinite]
    latencies = read_latencies(args.ping, target_time)
    latencies_isfinite = np.isfinite(latencies)
    latencies_finite = latencies[latencies_isfinite]

    f1, (a1, a2) = pp.subplots(2, sharex=True)
    colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    density_scatter(a1, durations.index, durations, s=1)
    a1.set_yscale("log")
    a1.set_ylabel("cURL time (ms)")

    density_scatter(a2, latencies.index, latencies, s=1)
    a2.set_yscale("log")
    a2.set_ylabel("Ping latency (ms)")

    f1.autofmt_xdate()
    f1.tight_layout()
    pp.show()

    f2, a3 = pp.subplots()
    bins = np.logspace(
        np.floor(np.log10(min(durations_finite.min(), latencies_finite.min()))),
        np.ceil(np.log10(max(durations_finite.max(), latencies_finite.max()))),
        100,
    )

    # Weight the samples according to the width in time.
    duration_weights = [
        t.total_seconds()
        for t in durations.index.to_series().diff()[durations_isfinite]
    ]
    latency_weights = [
        t.total_seconds()
        for t in latencies.index.to_series().diff()[latencies_isfinite]
    ]

    duration_cdf_samples, _, _ = a3.hist(
        durations_finite[1:],
        weights=duration_weights[1:],
        bins=bins,
        density=True,
        histtype="step",
        cumulative=True,
        label="cURL time",
    )
    latency_cdf_samples, _, _ = a3.hist(
        latencies_finite[1:],
        weights=latency_weights[1:],
        bins=bins,
        density=True,
        histtype="step",
        cumulative=True,
        label="Ping latency",
    )
    a3.set_xlabel("Time (ms)")
    a3.set_xscale("log")
    a3.set_xlim(bins[0], bins[-1])
    a3.legend()

    downtime = sum((durations_finite[1:] > 3500) * duration_weights[1:]) / sum(
        duration_weights[1:]
    )
    a3.set_title(f"Downtime: {downtime * 100:.2g}%")

    f2.tight_layout()
    pp.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
