#!/usr/bin/env python3

import argparse
import matplotlib as mpl
import matplotlib.pyplot as pp
import numpy as np
import itertools
import re
import sys

from datetime import datetime, timedelta
from pathlib import Path


def main(argv):
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
    args = parser.parse_args(argv[1:])

    latency_map = {}

    with args.file.open() as f:
        for line in map(str.strip, itertools.islice(f, 1, None)):
            time = datetime.strptime(
                re.match("^\[(.+)\]", line).group(1), "%Y-%m-%dT%H:%M:%S.%f"
            )
            if "Request timeout" in line:
                seq = int(re.search("\sicmp_seq (\d+)$", line).group(1))
                latency = np.inf
            else:
                seq = int(re.search("\sicmp_seq=(\d+)\s", line).group(1))
                latency = float(re.match(".+\stime=(.+) ms$", line).group(1))

            latency_map[seq] = (time, latency)

    seqs, tuples = zip(*sorted(latency_map.items()))
    times, latencies = zip(*tuples)

    # Change times to start time, rather than end time.
    times = [times[0] + s * timedelta(seconds=1) for s in seqs]

    seqs = np.array(seqs)
    times = np.array(times)
    latencies = np.array(latencies)
    outages = latencies > args.latency_threshold

    times_mid = (
        times[: -args.window] + (times[args.window :] - times[: -args.window]) / 2
    )
    outages_sum = np.cumsum(outages)
    outages_ave = (
        (outages_sum[args.window :] - outages_sum[: -args.window]) / args.window * 60
    )
    outages_ave[outages_ave == 0] = np.nan

    f, (a1, a2) = pp.subplots(2, sharex=True)
    a1.plot(times_mid, outages_ave)
    a2.scatter(times, latencies, s=1)
    a2.set_yscale("log")
    pp.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
