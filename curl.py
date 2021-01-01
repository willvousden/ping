import pandas as pd

from datetime import datetime
from pathlib import Path


def read_curl(file, target_time=None):
    path = Path(file)
    if target_time is None:
        target_time = datetime.fromtimestamp(0)

    names = ["timestamp"]
    names += [s[2:-1] for s in Path("curl-format.txt").read_text()[:-3].split()]
    series = (
        pd.read_csv(
            file,
            sep=" ",
            parse_dates=True,
            index_col=0,
            header=None,
            names=names,
            usecols=["timestamp", "time_total"],
        )["time_total"]
        * 1000
    )

    weights = [delta.total_seconds() for delta in series.index.to_series().diff()]
    durations = pd.DataFrame(
        {"ms": series.values, "weight": weights}, index=series.index
    )
    return durations[durations.index >= target_time]
