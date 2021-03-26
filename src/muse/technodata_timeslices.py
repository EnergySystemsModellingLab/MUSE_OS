from pathlib import Path
from typing import Hashable, List, Optional, Sequence, Text, Union, cast

import numpy as np
import pandas as pd
import xarray as xr


def read_technodata_timeslices(filename: Union[Text, Path]) -> xr.Dataset:
    from muse.readers import camel_to_snake

    csv = pd.read_csv(filename, float_precision="high", low_memory=False)
    csv = csv.rename(columns=camel_to_snake)
    data = csv[csv.process_name != "Unit"]
    ts = pd.MultiIndex.from_arrays(
        [
            data.process_name,
            data.region_name,
            [int(u) for u in data.time],
            data.month,
            data.day,
            data.hour,
        ],
        names=("technology", "region", "year", "month", "day", "hour"),
    )
    data.index = ts
    data.columns.name = "technodata"
    data.index.name = "technology"
    data = data.drop(
        ["process_name", "region_name", "time", "month", "day", "hour"], axis=1
    )

    data = data.apply(lambda x: pd.to_numeric(x, errors="ignore"), axis=0)
    result = xr.Dataset.from_dataframe(data.sort_index())
    return result
