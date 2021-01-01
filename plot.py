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


def cdf(durations, latencies, outage_mask):
    f2, a3 = pp.subplots()
    bins = np.logspace(
        np.floor(np.log10(min(durations["ms"].min(), latencies["ms"].min()))),
        np.ceil(np.log10(max(durations["ms"].max(), latencies["ms"].max()))),
        100,
    )

    duration_cdf_samples, _, _ = a3.hist(
        durations["ms"],
        weights=durations["weight"],
        bins=bins,
        density=True,
        histtype="step",
        cumulative=True,
        label="cURL time",
    )
    latency_cdf_samples, _, _ = a3.hist(
        latencies["ms"],
        weights=latencies["weight"],
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

    outage = sum(outage_mask * durations["weight"]) / sum(durations["weight"])
    a3.set_title(f"Outage: {outage * 100:.2g}%")

    f2.tight_layout()
    pp.show()


def scatter(durations, latencies, outage_mask):

    f, (a1, a2) = pp.subplots(2, sharex=True)
    colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    density_scatter(a1, durations.index, durations["ms"], s=1)
    a1.set_yscale("log")
    a1.set_ylabel("cURL time (ms)")

    density_scatter(a2, latencies.index, latencies["ms"], s=1)
    a2.set_yscale("log")
    a2.set_ylabel("Ping latency (ms)")

    f.autofmt_xdate()
    f.tight_layout()
    pp.show()


def time_of_day(durations, latencies, outage_mask):
    # Bin by time of day.
    hours = np.arange(25)

    def group(s):
        return s.groupby(by=lambda t: t.hour).sum().reindex(hours[:-1], fill_value=0)

    out_seconds = group(outage_mask * durations["weight"])
    all_seconds = group(durations["weight"])
    out_fraction = (out_seconds / all_seconds).fillna(0)

    f, a = pp.subplots()
    a.hist(hours[:-1] + 0.5, bins=hours, weights=out_fraction.values * 100)
    a.set_xlabel("Hour of day")
    a.set_ylabel("Outage (%)")
    a.set_xlim(0, 24)
    a.set_ylim(0, None)
    a.xaxis.set_ticks(hours[::3])
    a.xaxis.set_ticks(hours, minor=True)

    f.tight_layout()
    pp.show()


def main(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument("-p", "--ping", type=Path, required=True)
    parser.add_argument("-c", "--curl", type=Path, required=True)
    parser.add_argument(
        "-t",
        "--time",
        type=int,
        metavar="SECONDS",
        help="How many seconds of history to process",
    )
    subparsers = parser.add_subparsers(required=True)

    cdf_parser = subparsers.add_parser("cdf")
    cdf_parser.set_defaults(func=cdf)

    scatter_parser = subparsers.add_parser("scatter")
    scatter_parser.set_defaults(func=scatter)

    time_of_day_parser = subparsers.add_parser("time-of-day")
    time_of_day_parser.set_defaults(func=time_of_day)

    args = parser.parse_args(argv[1:])

    if args.time:
        target_time = datetime.now() - timedelta(seconds=args.time)
    else:
        target_time = datetime.fromtimestamp(0)

    durations = read_curl(args.curl, target_time)
    durations_isfinite = np.isfinite(durations["ms"]) & np.isfinite(durations["weight"])
    durations = durations[durations_isfinite]
    latencies = read_latencies(args.ping, target_time)
    latencies_isfinite = np.isfinite(latencies["ms"]) & np.isfinite(latencies["weight"])
    latencies = latencies[latencies_isfinite]

    outage_mask = (durations["ms"] > 3500) & (durations["ms"] < 30000)

    args.func(durations, latencies, outage_mask)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
