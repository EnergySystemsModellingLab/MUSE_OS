from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import (
    Any,
    Callable,
    NamedTuple,
    cast,
)

from xarray import DataArray, Dataset, zeros_like

from muse.outputs.cache import OutputCache
from muse.readers import read_initial_market
from muse.sectors import SECTORS_REGISTERED, AbstractSector, Sector
from muse.timeslices import broadcast_timeslice, drop_timeslice
from muse.utilities import future_propagation


class MCA:
    """Market Clearing Algorithm.

    The market clearing algorithm is the main object implementing the MUSE model. It is
    responsible for orchestrating  how the sectors are run, how they interface with one
    another, with the general market and the carbon market.
    """

    @classmethod
    def factory(cls, settings: str | Path | Mapping | Any) -> MCA:
        """Loads MCA from input settings and input files.

        Arguments:
            settings: namedtuple with the global MUSE input settings.

        Returns:
            The loaded MCA
        """
        from logging import getLogger

        from muse.outputs.mca import factory as ofactory
        from muse.readers import read_settings
        from muse.readers.toml import convert

        # Read settings
        if isinstance(settings, (str, Path)):
            settings = read_settings(settings)
        elif isinstance(settings, Mapping):
            settings = convert(settings)

        # Create the initial market
        market = (
            read_initial_market(
                projections_path=settings.global_input_files.projections,
                base_year_export_path=getattr(
                    settings.global_input_files, "base_year_export", None
                ),
                base_year_import_path=getattr(
                    settings.global_input_files, "base_year_import", None
                ),
                currency=getattr(settings, "currency", None),
            )
            .sel(region=settings.regions)
            .interp(year=settings.time_framework, method=settings.interpolation_mode)
        )
        market["supply"] = drop_timeslice(zeros_like(market.exports))
        market["consumption"] = drop_timeslice(zeros_like(market.exports))

        # Create the sectors
        sectors = []
        for sector in settings.sectors.list:
            kind = getattr(settings.sectors, sector).type
            sectors.append(SECTORS_REGISTERED[kind](sector, settings))
            getLogger(__name__).info(f"Created sector {sector}")

        # Check that sector commodities are disjoint
        commodities = [c for s in sectors for c in s.commodities]
        duplicates = [c for c in set(commodities) if commodities.count(c) > 1]
        if duplicates:
            getLogger(__name__).critical(
                f"Commodities {duplicates} are outputted by multiple sectors."
            )

        # Create the outputs
        outputs = ofactory(*getattr(settings, "outputs", []))
        outputs_cache = OutputCache(
            *getattr(settings, "outputs_cache", []), sectors=sectors
        )

        # Global settings for MCA
        extras = {
            "regions",
            "log_level",
            "interpolation_mode",
            "timeslices",
            "root",
            "plugins",
            "outputs",
            "outputs_cache",
            "currency",
        }
        global_kw = {
            k: v
            for k, v in settings._asdict().items()
            if not hasattr(v, "_asdict") and k not in extras
        }

        # Carbon budget settings
        carbon_kw = {
            k: v._asdict() if hasattr(v, "_asdict") else v
            for k, v in settings.carbon_budget_control._asdict().items()
        }
        for key in {"budget", "commodities", "method"}:
            if key in carbon_kw:
                carbon_kw[f"carbon_{key}"] = carbon_kw.pop(key)

        # Create the MCA
        return cls(
            sectors=sectors,
            market=market,
            outputs=outputs,
            outputs_cache=outputs_cache,
            **global_kw,
            **carbon_kw,
        )

    def __init__(
        self,
        sectors: list[AbstractSector],
        market: Dataset,
        outputs: Callable[[list[AbstractSector], Dataset], Any] | None = None,
        outputs_cache: OutputCache | None = None,
        time_framework: Sequence[int] = list(range(2010, 2100, 10)),
        equilibrium: bool = True,
        equilibrium_variable: str = "demand",
        maximum_iterations: int = 100,
        tolerance: float = 0.1,
        tolerance_unmet_demand: float = -0.1,
        excluded_commodities: Sequence[str] | None = None,
        carbon_budget: Sequence | None = None,
        carbon_commodities: Sequence[str] | None = None,
        debug: bool = False,
        control_undershoot: bool = False,
        control_overshoot: bool = False,
        carbon_method: str = "bisection",
        method_options: Mapping | None = None,
    ):
        """Market clearing algorithm class which rules the whole MUSE."""
        from logging import getLogger

        from numpy import array
        from xarray import DataArray

        from muse.carbon_budget import CARBON_BUDGET_METHODS
        from muse.outputs.mca import factory as ofactory

        getLogger(__name__).info("MCA Initialisation")

        self.sectors: list[AbstractSector] = list(sectors)
        self.market = market

        # Simulation flow parameters
        self.time_framework = array(time_framework)
        self.equilibrium = equilibrium
        self.equilibrium_variable = equilibrium_variable
        self.maximum_iterations = maximum_iterations
        self.tolerance = tolerance
        self.tolerance_unmet_demand = tolerance_unmet_demand
        if excluded_commodities:
            self.excluded_commodities = excluded_commodities
        else:
            self.excluded_commodities = []

        # Carbon budget parameters
        if isinstance(carbon_budget, DataArray) and "year" in carbon_budget.dims:
            self.carbon_budget: DataArray = carbon_budget.sel(year=time_framework)
        elif isinstance(carbon_budget, DataArray):
            self.carbon_budget = carbon_budget
        elif carbon_budget is not None and len(carbon_budget) > 0:
            assert len(carbon_budget) == len(time_framework)
            self.carbon_budget = DataArray(
                carbon_budget, dims="year", coords={"year": time_framework}
            )
        else:
            self.carbon_budget = DataArray([], dims="year")
        self.carbon_commodities = carbon_commodities or []
        self.debug = debug
        self.control_undershoot = control_undershoot
        self.control_overshoot = control_overshoot
        self.carbon_method = CARBON_BUDGET_METHODS[carbon_method]
        self.method_options = method_options or {}
        self.outputs = outputs or ofactory()
        self.outputs_cache = outputs_cache or OutputCache()

    def find_equilibrium(
        self,
        market: Dataset,
    ) -> FindEquilibriumResults:
        """Specialised version of the find_equilibrium function.

        Arguments:
            market: Commodities market, with the prices, supply, consumption and demand.

        Returns:
            A tuple with the updated market (prices, supply, consumption and demand) and
            sector.
        """
        return find_equilibrium(
            market=market,
            sectors=self.sectors,
            maxiter=self.maximum_iterations,
            tol=self.tolerance,
            equilibrium_variable=self.equilibrium_variable,
            tol_unmet_demand=self.tolerance_unmet_demand,
            excluded_commodities=self.excluded_commodities,
            equilibrium=self.equilibrium,
        )

    def update_carbon_budget(self, market: Dataset, year_idx: int) -> float:
        """Specialised version of the update_carbon_budget function.

        Arguments:
            market: Commodities market, with the prices, supply, consumption and demand.
            year_idx: Index of the year of interest.

        Returns:
            An updated market with prices, supply, consumption and demand.
        """
        from muse.carbon_budget import update_carbon_budget

        emission = (
            market.supply.sel(year=market.year[-1], commodity=self.carbon_commodities)
            .sum()
            .values
        )

        return update_carbon_budget(
            cast(Sequence[float], self.carbon_budget),
            emission,
            year_idx,
            self.control_overshoot,
            self.control_undershoot,
        )

    def update_carbon_price(self, market) -> float:
        """Calculates the updated carbon price.

        Arguments:
            market: Market with the prices, supply, consumption and demand.

        Returns:
            The new carbon price.
        """
        new_carbon_price = self.carbon_method(  # type: ignore
            market,
            self.find_equilibrium,
            self.carbon_budget,
            self.carbon_commodities,
            **self.method_options,
        )

        return new_carbon_price

    def run(self) -> None:
        """Initiates the calculation, starting with the loop over years.

        This method starts the main MUSE loop, going over the years of the simulation.
        Internally, it runs the carbon budget loop, which updates the carbon prices, if
        needed, and the equilibrium loop, which tries to reach an equilibrium between
        prices, demand and supply.

        Returns:
            None
        """
        from logging import getLogger

        from xarray import DataArray

        nyear = len(self.time_framework) - 1
        check_carbon_budget = len(self.carbon_budget) and len(self.carbon_commodities)
        shoots = self.control_undershoot or self.control_overshoot
        variables = ["supply", "consumption", "prices"]

        for year_idx in range(nyear):
            current_year, investment_year = self.time_framework[year_idx : year_idx + 2]
            getLogger(__name__).info(
                f"Running simulation years {current_year} to {investment_year}"
            )
            new_market = self.market[variables].sel(
                year=[current_year, investment_year]
            )
            assert isinstance(new_market, Dataset)
            new_market.supply[:] = 0
            new_market.consumption[:] = 0

            # If we need to account for the carbon budget, we do it now.
            if check_carbon_budget:
                getLogger(__name__).info(
                    f"Updating carbon price for year {investment_year}"
                )
                new_price = self.update_carbon_price(new_market)
                future_price = DataArray(new_price, coords=dict(year=investment_year))
                new_market.prices.loc[dict(commodity=self.carbon_commodities)] = (
                    future_propagation(
                        new_market.prices.sel(commodity=self.carbon_commodities),
                        broadcast_timeslice(future_price),
                    )
                )

            # Solve the market
            _, new_market, self.sectors = self.find_equilibrium(new_market)

            # Save sector outputs for the investment year
            # In the first iteration we also save outputs for the first year
            for sector in self.sectors:
                if type(sector) is Sector:
                    if year_idx == 0:
                        sector.save_outputs(current_year)
                    sector.save_outputs(investment_year)

            # If we need to account for the carbon budget, we might need to change
            # the budget for the future, too.
            if check_carbon_budget and shoots and year_idx < nyear - 2:
                self.carbon_budget[year_idx + 2] = self.update_carbon_budget(
                    new_market, year_idx
                )

            # Update the market
            dims = {i: new_market[i] for i in new_market.dims}
            self.market.supply.loc[dims] = new_market.supply
            self.market.consumption.loc[dims] = new_market.consumption

            # Update prices
            dims = {i: new_market[i] for i in new_market.prices.dims if i != "year"}
            self.market.prices.loc[dims] = future_propagation(
                self.market.prices.sel(dims),
                new_market.prices.sel(year=investment_year),
            )

            # Global outputs for the investment year
            # In the first iteration we also save outputs for the first year
            if year_idx == 0:
                self.outputs(self.market, self.sectors, year=current_year)
            self.outputs(self.market, self.sectors, year=investment_year)
            self.outputs_cache.consolidate_cache(year=investment_year)

            getLogger(__name__).info(
                f"Finished simulation period {current_year} to {investment_year} "
                f"({year_idx + 1}/{nyear})!"
            )


class SingleYearIterationResult(NamedTuple):
    """Result of iterating over sectors for a year.

    Convenience tuple naming naming the return values from  of
    :py:func:`single_year_iteration`.
    """

    market: Dataset
    sectors: list[AbstractSector]
    updated_prices: DataArray


def single_year_iteration(
    market: Dataset, sectors: list[AbstractSector]
) -> SingleYearIterationResult:
    """Runs one iteration of the sectors (runs each sector once).

    Arguments:
        market: An initial market with prices, supply, consumption.
        sectors: A list of the sectors participating in the simulation.

    Returns:
        A tuple with the new market, sectors and updated prices.
    """
    from copy import deepcopy

    sectors = deepcopy(sectors)
    market = market.copy(deep=True)

    # New prices for the investment year
    investment_year = market.year[1]
    updated_prices = market.prices.sel(year=investment_year)

    for sector in sectors:
        # Solve the sector
        sector_market = sector.next(market[["supply", "consumption", "prices"]])

        # Reindex to add back commodities that are not relevant to the sector
        sector_market = sector_market.reindex_like(market, fill_value=0)

        # Update market supply and consumption
        market["consumption"] = drop_timeslice(
            market.consumption + sector_market.consumption
        )
        market["supply"] = drop_timeslice(market.supply + sector_market.supply)

        # Update market prices
        # We only do this for the commodities that the sector is in charge of producing
        # And only for regions/timeslices with >0 production in the investment year
        supply = sector_market.supply.sel(year=investment_year)
        supply = supply.where(supply.commodity.isin(sector.commodities), 0)
        updated_prices = updated_prices.where(
            supply == 0, sector_market.costs.sel(year=investment_year)
        )

    return SingleYearIterationResult(market, sectors, updated_prices)


class FindEquilibriumResults(NamedTuple):
    """Result of find equilibrium."""

    converged: bool
    market: Dataset
    sectors: list[AbstractSector]


def find_equilibrium(
    market: Dataset,
    sectors: list[AbstractSector],
    maxiter: int = 3,
    tol: float = 0.1,
    equilibrium_variable: str = "demand",
    tol_unmet_demand: float = -0.1,
    excluded_commodities: Sequence | None = None,
    equilibrium: bool = True,
) -> FindEquilibriumResults:
    """Runs the equilibrium loop.

    If convergence is reached, then the function returns the new market. If the maximum
    number of iterations are reached, then a warning issued in the log and the function
    returns with the current status.

    Arguments:
        market: Commodities market, with the prices, supply, consumption and demand.
        sectors: A list of the sectors participating in the simulation.
        maxiter: Maximum number of iterations.
        tol: Tolerance for reaching equilibrium.
        equilibrium_variable: Variable to use to calculate the equilibrium condition.
        tol_unmet_demand: Tolerance for the unmet demand.
        excluded_commodities: Commodities to be excluded in check_demand_fulfillment
        equilibrium: if equilibrium should be reached. Useful to testing.

    Returns:
        A tuple with the updated market (prices, supply, consumption and demand),
        sectors, and convergence status.
    """
    from logging import getLogger

    from numpy import ones

    market = market.copy(deep=True)
    current_year, investment_year = market.year

    if excluded_commodities:
        included = ~market.commodity.isin(excluded_commodities)
    else:
        included = ones(len(market.commodity), dtype=bool)

    converged = False
    iteration = 0
    while iteration < maxiter and not converged:
        prior_market = market.copy(deep=True)
        market.consumption[:] = 0.0
        market.supply[:] = 0.0
        market, equilibrium_sectors, updated_prices = single_year_iteration(
            market, sectors
        )

        if maxiter == 1 or not equilibrium:
            converged = True
            break

        # Update prices
        market["prices"].loc[dict(year=investment_year)] = updated_prices

        # Check convergence
        converged = check_equilibrium(
            market.sel(commodity=included),
            prior_market.sel(commodity=included),
            tol,
            equilibrium_variable,
            investment_year,
        )
        iteration += 1

    if not converged:
        msg = (
            f"CONVERGENCE ERROR: Maximum number of iterations ({maxiter}) reached "
            f"in year {int(current_year)}"
        )
        getLogger(__name__).critical(msg)

    # Check that demand is fulfilled (raises a warning if not)
    check_demand_fulfillment(market.sel(commodity=included), tol_unmet_demand)

    return FindEquilibriumResults(converged, market, equilibrium_sectors)


def check_demand_fulfillment(market: Dataset, tol: float) -> bool:
    """Checks if the supply will fulfill all the demand in the future.

    If it does not, it logs a warning.

    Arguments:
        market: Commodities market, with the prices, supply, consumption and demand.
        tol: Tolerance for the unmet demand.

    Returns:
        True if the supply fulfils the demand; False otherwise
    """
    from logging import getLogger

    future = market.year[-1].item()
    delta = (market.supply.sum("timeslice") - market.consumption.sum("timeslice")).sel(
        year=future
    )
    unmet = (delta < tol).any([u for u in delta.dims if u != "commodity"])

    if unmet.any():
        commodities = ", ".join(unmet.commodity.sel(commodity=unmet.values).values)
        msg = (
            f"Consumption exceeds supply in the year {future} for the following "
            f"commodities: {commodities} "
        )
        getLogger(__name__).warning(msg)

        return False

    return True


def check_equilibrium(
    market: Dataset,
    int_market: Dataset,
    tolerance: float,
    equilibrium_variable: str,
    year: int | None = None,
) -> bool:
    """Checks if equilibrium has been reached.

    This function checks if the difference in either the demand or the prices between
    iterations if smaller than certain tolerance. If is, then it is assumed that
    the process has converged.

    Arguments:
        market: The market values in this iteration.
        int_market: The market values in the previous iteration.
        tolerance: Tolerance for reaching equilibrium.
        equilibrium_variable: Variable to use to calculate the equilibrium condition.
        year: year for which to check changes. Default to minimum year in market.

    Returns:
        True if converged, False otherwise.
    """
    from numpy import abs

    if year is None:
        year = market.year.min()

    if equilibrium_variable == "demand":
        delta = (
            market.consumption.sel(year=year)
            - market.supply.sel(year=year)
            - int_market.consumption.sel(year=year)
            + int_market.supply.sel(year=year)
        )
    else:
        delta = market.prices.sel(year=year) - int_market.prices.sel(year=year)

    return bool((abs(delta) < tolerance).all())
