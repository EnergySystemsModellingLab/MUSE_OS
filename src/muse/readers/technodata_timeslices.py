from pathlib import Path
from typing import Hashable, List, Optional, Sequence, Text, Union, cast

import numpy as np
import pandas as pd
import xarray as xr


def read_technodata_timeslices(filename: Union[Text, Path]) -> xr.Dataset:
    csv = pd.read_csv(filename, float_precision="high", low_memory=False)
    data = csv[csv.ProcessName != "Unit"]
