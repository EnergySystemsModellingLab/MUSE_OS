from typing import Any, Callable, List, Mapping, MutableMapping, Optional, Text, Union

import pandas as pd
import xarray as xr
from mypy_extensions import KwArg

from muse.registration import registrator
from muse.outputs.sector import register_output_quantity
from muse.outputs.sector import market_quantity


@register_output_quantity
def supply_timeslice(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
    rounding: int = 4,
) -> xr.DataArray:
    """Current supply."""
    print(market)
    market = market.reset_index("timeslice")
    result = (
        market_quantity(market.supply, sum_over=sum_over, drop=drop)
        .rename("supply")
        .to_dataframe()
        .round(rounding)
    )
    return result[result.supply != 0]
