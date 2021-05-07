#!/usr/bin/env python3

import argparse
import sys

from pathlib import Path
from pinglib.format.ping import get_record_count, tail_records
from pinglib.plotting.common import density_scatter


def main(argv):
    import matplotlib as mpl
    import matplotlib.pyplot as pp
    import numpy as np
    import pandas as pd

    def get_latencies(path, count=None):
        path = Path(path)
        if count is None:
            count = get_record_count(path)
        times = np.empty(count, dtype="datetime64[ms]")
        latencies = np.empty(count, dtype="float64")
        for i, (time, latency) in enumerate(tail_records(path, count)):
            times[i] = time
            latencies[i] = latency

        weights = [delta.total_seconds() for delta in pd.Series(times).diff()]
        return pd.DataFrame({"ms": latencies, "weight": weights}, index=times)

    parser = argparse.ArgumentParser()
    parser.add_argument("file", type=Path)
    parser.add_argument(
        "-l",
        "--latency-threshold",
        type=float,
        default=1000,
        metavar="MILLISECONDS",
        help="The minimum latency, in milliseconds, for a ping to be considered an outage.",
    )
    parser.add_argument(
        "-w",
        "--window",
        type=int,
        default=300,
        metavar="SECONDS",
        help="The moving average window size, in seconds, for calculating the outage rate.",
    )
    parser.add_argument(
        "-c",
        "--count",
        type=int,
        metavar="PINGS",
        help="How many ping records of history to process.",
    )
    args = parser.parse_args(argv[1:])

    latencies = get_latencies(args.file, args.count)
    outages = latencies["ms"] > args.latency_threshold

    outages_ave = outages.rolling(args.window, center=True).mean() * 60
    outages_ave[outages_ave == 0] = np.nan

    f, (a1, a2) = pp.subplots(2, sharex=True)
    colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    a1.plot(outages_ave.index, outages_ave)
    a1.set_ylabel("Outage rate (s/min)")

    quantile = 0.99
    finite = np.isfinite(latencies["ms"])
    density_scatter(
        a2,
        latencies.loc[finite].index,
        latencies["ms"].loc[finite],
        s=1,
        label="Ping latency",
    )
    a2.plot(
        latencies.loc[finite].index,
        latencies["ms"]
        .loc[finite]
        .rolling(args.window, center=True)
        .quantile(quantile),
        c=colours[1],
        label=f"{quantile} quantile",
    )
    a2.legend()
    a2.set_yscale("log")
    a2.set_ylabel("Latency (ms)")

    f.autofmt_xdate()
    f.tight_layout()
    pp.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
