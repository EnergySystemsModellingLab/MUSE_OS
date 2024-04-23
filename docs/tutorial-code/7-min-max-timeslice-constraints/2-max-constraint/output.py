from typing import List, Optional, Text

import pandas as pd
import xarray as xr
from muse.outputs.sector import market_quantity, register_output_quantity


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
    market = market.reset_index("timeslice")
    result = (
        market_quantity(market.supply, sum_over=sum_over, drop=drop)
        .rename("supply")
        .to_dataframe()
        .round(rounding)
    )
    return pd.DataFrame(result[result.supply != 0]).reset_index()


@register_output_quantity
def consumption_timeslice(
    market: xr.Dataset,
    capacity: xr.DataArray,
    technologies: xr.Dataset,
    sum_over: Optional[List[Text]] = None,
    drop: Optional[List[Text]] = None,
    rounding: int = 4,
) -> xr.DataArray:
    """Current consumption."""
    market = market.reset_index("timeslice")
    result = (
        market_quantity(market.consumption, sum_over=sum_over, drop=drop)
        .rename("consumption")
        .to_dataframe()
        .round(rounding)
    )
    return pd.DataFrame(result[result.consumption != 0]).reset_index()
