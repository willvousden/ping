#!/usr/bin/env python3

import argparse
import io
import itertools
import matplotlib as mpl
import matplotlib.pyplot as pp
import numpy as np
import pandas as pd
import re
import struct
import sys

from datetime import datetime, timedelta
from pathlib import Path


RECORD_STRUCT = struct.Struct("df")


def parse_lines(lines):
    seq_offset = 0
    seq_prev = -1
    for line in map(str.strip, lines):
        time = datetime.strptime(
            re.match(r"^\[(.+)\]", line).group(1), "%Y-%m-%dT%H:%M:%S.%f"
        )

        if "Request timeout" in line:
            seq = int(re.search(r"\sicmp_seq (\d+)$", line).group(1))
            latency = float("inf")
        elif "64 bytes" in line:
            seq = int(re.search(r"\sicmp_seq=(\d+)\s", line).group(1))
            latency = float(re.match(r".+\stime=(.+) ms$", line).group(1))
        else:
            continue

        # The ICMP sequence number eventually wraps around.
        if seq == 0:
            seq_offset = seq_prev + 1

        yield time, seq, latency


def pack_record(time, latency):
    return RECORD_STRUCT.pack(time.timestamp(), latency)


def unpack_record(buffer):
    return RECORD_STRUCT.unpack(buffer)


def tail_records(path, count):
    path = Path(path)
    with path.open("rb") as f:
        f.seek(-count * RECORD_STRUCT.size, io.SEEK_END)
        while True:
            buffer = f.read(RECORD_STRUCT.size)
            if buffer:
                yield unpack_record(buffer)
            else:
                break


def read_latencies(file, target_time=None):
    file = Path(file)
    if target_time is None:
        target_time = datetime.fromtimestamp(0)

    latency_map = {}
    seq_offset = 0
    seq_prev = -1
    with file.open() as f:
        for time, seq, latency in parse_lines(itertools.islice(f, 1, None)):
            seq += seq_offset
            latency_map[seq] = (time, latency)
            seq_prev = seq

    _, tuples = zip(*sorted(latency_map.items()))
    times, latencies = zip(*tuples)
    weights = [delta.total_seconds() for delta in pd.Series(times).diff()]
    return pd.DataFrame({"ms": latencies, "weight": weights}, index=times)


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

    latencies = read_latencies(args.file, target_time)
    outages = latencies > args.latency_threshold

    outages_ave = outages.rolling(args.window, center=True).mean() * 60
    outages_ave[outages_ave == 0] = np.nan

    f, (a1, a2) = pp.subplots(2, sharex=True)
    colours = mpl.rcParams["axes.prop_cycle"].by_key()["color"]

    a1.plot(outages_ave.index, outages_ave)
    a1.set_ylabel("Outage rate (s/min)")

    finite = np.isfinite(latencies)
    a2.scatter(latencies.loc[finite].index, latencies.loc[finite], s=1)
    a2.plot(
        latencies.loc[finite].index,
        latencies.loc[finite].rolling(args.window, center=True).quantile(0.99),
        c=colours[1],
    )
    a2.set_yscale("log")
    a2.set_ylabel("Latency (ms)")

    f.autofmt_xdate()
    f.tight_layout()
    pp.show()


if __name__ == "__main__":
    sys.exit(main(sys.argv))
