#!/usr/bin/env python3

import argparse
import io
import itertools
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
    time, latency = RECORD_STRUCT.unpack(buffer)
    return datetime.fromtimestamp(time), latency


def tail_records(path, count):
    path = Path(path)
    with path.open("rb") as f:
        f.seek(-count * RECORD_STRUCT.size, io.SEEK_END)
        for i in range(count):
            buffer = f.read(RECORD_STRUCT.size)
            if buffer:
                yield unpack_record(buffer)
            else:
                break


def main(argv):
    import matplotlib as mpl
    import matplotlib.pyplot as pp
    import numpy as np
    import pandas as pd

    def get_latencies(path, count=None):
        path = Path(path)
        if count is None:
            count = path.stat().st_size // RECORD_STRUCT.size
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
    a2.scatter(
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
