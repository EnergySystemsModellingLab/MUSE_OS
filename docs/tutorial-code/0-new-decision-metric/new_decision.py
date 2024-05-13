from typing import Any

from muse.decisions import register_decision
from xarray import DataArray, Dataset


@register_decision
def median_objective(objectives: Dataset, parameters: Any, **kwargs) -> DataArray:
    from xarray import concat

    allobjectives = concat(objectives.data_vars.values(), dim="concat_var")
    return allobjectives.median(set(allobjectives.dims) - {"asset", "replacement"})
