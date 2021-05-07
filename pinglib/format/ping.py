import io
import numpy as np
import pandas as pd
import re
import struct

from datetime import datetime
from pathlib import Path


RECORD_STRUCT = struct.Struct("df")
RECORD_NUMPY_DTYPE = np.dtype([("timestamp", np.float64), ("ms", np.float32)])


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


def get_record_count(path):
    path = Path(path)
    return path.stat().st_size // RECORD_STRUCT.size


def read_to_pandas(path):
    data = np.fromfile(str(path), dtype=RECORD_NUMPY_DTYPE)
    df = pd.DataFrame(data)
    df["weight"] = df["timestamp"].diff()
    df.index = pd.to_datetime(df["timestamp"], unit="s")
    del df["timestamp"]
    return df
