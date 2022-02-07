"""Output cached quantities

Functions that output the state of diverse quantities at intermediate steps of the
calculation.

The core of the method is the OutputCache class that initiated by the MCA with input
parameters defined in the TOML file (much like the existing 'output' options),
enables a channel to "listen" for data to be cached and, after each period, saved into
disk via the 'consolidate_cache' method.

Anywhere in the code, you can write:

.. code-block:: python

    pub.sendMessage("cache_quantity", quantity=quantity_name, data=some_data)

If the quantity has been set as something to cache, the data will be stored and,
eventually, save to disk after - possibly - agregating the data and remove those entries
corresponding to non-convergent investment attempts. This process of cleaning and
aggregation is quantity specific.
"""
from __future__ import annotations

from typing import List, Mapping, Text

from pubsub import pub
import xarray as xr


class OutputCache:
    def __init__(self, parameters: Mapping) -> None:
        self.param = parameters
        self.to_save: Mapping[str, List[xr.DataArray]] = {
            p["quantity"]: [] for p in self.param
        }
        pub.subscribe(self.cache, "cache_quantity")

    def cache(self, quantity: Text, data: xr.DataArray) -> None:
        """Caches the data into memory for the given quantity.

        Args:
            quantity (str): The quantity this data relates to.
            data (xr.DataArray): The data to be cache.
        """
        if quantity not in self.to_save:
            return
        self.to_save[quantity].append(data.copy())

    def consolidate_cache(self) -> None:
        """Save the cached data into disk and flushes cache.

        This method is meant to be called after each time period in the main loop of the
        MCA, at the same time that market quantities are saved.

        TODO: Actually implement the "save" part, ideally re-using most of the
        infrastructure for saving stuff already in place.
        """
        self.to_save = {p["quantity"]: [] for p in self.param}
