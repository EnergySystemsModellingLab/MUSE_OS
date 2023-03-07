from __future__ import annotations

from pathlib import Path
from typing import (
    Any,
    Callable,
    List,
    Mapping,
    NamedTuple,
    Optional,
    Sequence,
    Text,
    Union,
    cast,
)

from xarray import Dataset, zeros_like

from muse.outputs.cache import OutputCache
from muse.readers import read_initial_market
from muse.sectors import SECTORS_REGISTERED, AbstractSector


class MCA(object):
    """Market Clearing Algorithm.

    The market clearing algorithm is the main object implementing the MUSE model. It is
    responsible for orchestrating  how the sectors are run, how they interface with one
    another, with the general market and the carbon market.
    """

    @classmethod
    def factory(cls, settings: Union[Text, Path, Mapping, Any]) -> MCA:
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

        if isinstance(settings, (Text, Path)):
            settings = read_settings(settings)  # type: ignore
        elif isinstance(settings, Mapping):
            settings = convert(settings)
        settings = cast(Any, settings)
        # We create the initial market
        market = (
            read_initial_market(
                settings.global_input_files.projections,
                base_year_export=getattr(
                    settings.global_input_files, "base_year_export", None
                ),
                base_year_import=getattr(
                    settings.global_input_files, "base_year_import", None
                ),
                timeslices=settings.timeslices,
            ).sel(region=settings.regions)
        ).interp(year=settings.time_framework, method=settings.interpolation_mode)

        market["supply"] = zeros_like(market.exports)
        market["consumption"] = zeros_like(market.exports)

        # We create the sectors
        sectors = []
        for sector in settings.sectors.list:
            kind = getattr(settings.sectors, sector).type
            sectors.append(SECTORS_REGISTERED[kind](sector, settings))
            getLogger(__name__).info(f"Created sector {sector}")

        outputs = ofactory(*getattr(settings, "outputs", []))
        outputs_cache = OutputCache(
            *getattr(settings, "outputs_cache", []), sectors=sectors
        )

        extras = {
            "foresight",
            "regions",
            "interest_rate",
            "log_level",
            "interpolation_mode",
            "timeslices",
            "root",
            "plugins",
            "outputs",
            "outputs_cache",
        }
        global_kw = {
            k: v
            for k, v in settings._asdict().items()
            if not hasattr(v, "_asdict") and k not in extras
        }
        if "equilibrium" in global_kw:
            global_kw["equilibrium"] = global_kw.pop("equilibrium")
        carbon_kw = {
            k: v._asdict() if hasattr(v, "_asdict") else v
            for k, v in settings.carbon_budget_control._asdict().items()
        }
        for key in {"budget", "commodities", "method"}:
            carbon_kw[f"carbon_{key}"] = carbon_kw[key]
            carbon_kw.pop(key)
        return cls(
            sectors=sectors,
            market=market,
            outputs=outputs,  # type: ignore
            outputs_cache=outputs_cache,  # type: ignore
            **global_kw,
            **carbon_kw,
        )

    def __init__(
        self,
        sectors: List[AbstractSector],
        market: Dataset,
        outputs: Optional[Callable[[List[AbstractSector], Dataset], Any]] = None,
        outputs_cache: Optional[OutputCache] = None,
        time_framework: Sequence[int] = list(range(2010, 2100, 10)),
        equilibrium: bool = True,
        equilibrium_variable: Text = "demand",
        maximum_iterations: int = 3,
        tolerance: float = 0.1,
        tolerance_unmet_demand: float = -0.1,
        excluded_commodities: Optional[Sequence[Text]] = None,
        carbon_budget: Optional[Sequence] = None,
        carbon_price: Optional[Sequence] = None,
        carbon_commodities: Optional[Sequence[Text]] = None,
        debug: bool = False,
        control_undershoot: bool = True,
        control_overshoot: bool = True,
        carbon_method: Text = "fitting",
        method_options: Optional[Mapping] = None,
    ):
        """Market clearing algorithm class which rules the whole MUSE."""
        from logging import getLogger

        from numpy import array
        from xarray import DataArray

        from muse.carbon_budget import CARBON_BUDGET_METHODS
        from muse.outputs.mca import factory as ofactory

        getLogger(__name__).info("MCA Initialisation")

        self.sectors: List[AbstractSector] = list(sectors)
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
            self.carbon_budget: DataArray = (
                carbon_budget.interp(year=time_framework).ffill("year").bfill("year")
            )
        elif isinstance(carbon_budget, DataArray):
            self.carbon_budget = carbon_budget
        elif carbon_budget is not None and len(carbon_budget) > 0:
            assert len(carbon_budget) == len(time_framework)
            self.carbon_budget = DataArray(
                carbon_budget, dims="year", coords={"year": time_framework}
            )
        else:
            self.carbon_budget = DataArray([], dims="year")
        self.carbon_price = (
            carbon_price if carbon_price is not None else zeros_like(self.carbon_budget)
        )
        self.carbon_commodities = (
            carbon_commodities if carbon_commodities is not None else []
        )
        self.debug = debug
        self.control_undershoot = control_undershoot
        self.control_overshoot = control_overshoot
        self.carbon_method = CARBON_BUDGET_METHODS[carbon_method]
        self.method_options = method_options
        self.outputs = ofactory() if outputs is None else outputs
        self.outputs_cache = OutputCache() if outputs_cache is None else outputs_cache

    def find_equilibrium(
        self,
        market: Dataset,
        sectors: Optional[List[AbstractSector]] = None,
        maxiter: Optional[int] = None,
    ) -> FindEquilibriumResults:
        """Specialised version of the find_equilibrium function.

        Arguments:
            market: Commodities market, with the prices, supply, consumption and demand.

        Returns:
            A tuple with the updated market (prices, supply, consumption and demand) and
            sector.
        """
        maxiter = self.maximum_iterations if not maxiter else maxiter
        return find_equilibrium(
            market=market,
            sectors=self.sectors if sectors is None else sectors,
            maxiter=maxiter,
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

    def update_carbon_price(self, market) -> Optional[float]:
        """Calculates the updated carbon price, if required.

        If the emission calculated for the next time period is larger than the
        limit, then the carbon price needs to be updated to ensure that whatever the
        sectors do, the carbon budget limit is not exceeded.

        Arguments:
            market: Market, with the prices, supply, consumption and demand.

        Returns:
            The new carbon price or None
        """
        from numpy import median

        future = market.year[-1]

        market, _ = single_year_iteration(market, self.sectors)

        threshold = self.carbon_budget.interp(
            year=future, kwargs=dict(fill_value=self.carbon_budget.isel(year=-1).values)
        ).values
        emissions = (
            market.supply.sel(year=future, commodity=self.carbon_commodities)
            .sum(["region", "timeslice", "commodity"])
            .values
        )

        # Future emissions are OK, so we move on
        cp = median(market.prices.sel(commodity=self.carbon_commodities, year=future))
        if emissions < threshold and not self.debug and cp == 0.0:
            return None

        new_carbon_price = self.carbon_method(  # type: ignore
            market,
            self.sectors,
            self.find_equilibrium,
            self.carbon_budget,
            self.carbon_price,
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

        from numpy import where
        from xarray import DataArray

        from muse.utilities import future_propagation

        _, self.sectors, hist_years = self.calibrate_legacy_sectors()
        if len(hist_years) > 0:
            hist = where(self.time_framework <= hist_years[-1])[0]
            start = hist[-1]

        else:
            start = -1

        nyear = len(self.time_framework) - 1
        check_carbon_budget = len(self.carbon_budget) and len(self.carbon_commodities)
        shoots = self.control_undershoot or self.control_overshoot
        variables = ["supply", "consumption", "prices"]

        for year_idx in range(start + 1, nyear):
            years = self.time_framework[year_idx : year_idx + 2]
            getLogger(__name__).info(f"Running simulation year {years[0]}...")
            new_market = self.market[variables].sel(year=years)
            assert isinstance(new_market, Dataset)
            new_market.supply[:] = 0
            new_market.consumption[:] = 0

            # If we need to account for the carbon budget, we do it now.
            if check_carbon_budget:
                new_price = self.update_carbon_price(new_market)
                if new_price is not None:
                    future_price = DataArray(new_price, coords=dict(year=years[1]))

                    new_market.prices.loc[
                        dict(commodity=self.carbon_commodities)
                    ] = future_propagation(
                        new_market.prices.sel(commodity=self.carbon_commodities),
                        future_price,
                    )
                    self.carbon_price = future_propagation(
                        self.carbon_price, future_price
                    )

            _, new_market, self.sectors = self.find_equilibrium(new_market)

            # If we need to account for the carbon budget, we might need to change
            # the budget for the future, too.
            if check_carbon_budget and shoots and year_idx < nyear - 2:
                self.carbon_budget[year_idx + 2] = self.update_carbon_budget(
                    new_market, year_idx
                )

            dims = {i: new_market[i] for i in new_market.dims}
            self.market.supply.loc[dims] = new_market.supply
            self.market.consumption.loc[dims] = new_market.consumption

            dims = {i: new_market[i] for i in new_market.prices.dims if i != "year"}
            self.market.prices.loc[dims] = future_propagation(
                self.market.prices.sel(dims), new_market.prices.sel(year=years[1])
            )

            self.outputs(
                self.market, self.sectors, year=self.time_framework[year_idx]
            )  # type: ignore
            self.outputs_cache.consolidate_cache(year=self.time_framework[year_idx])
            getLogger(__name__).info(f"Finish simulation year {years[0]}!")

    def calibrate_legacy_sectors(self):
        """Run a calibration step in the lagacy sectors
        Run historical years
        """
        from copy import deepcopy
        from logging import getLogger

        from numpy import clip, where

        hist_years = []
        if len([s for s in self.sectors if "LegacySector" in str(type(s))]) == 0:
            return None, self.sectors, hist_years

        sectors = []
        idx = []
        for i, s in enumerate(self.sectors):
            if "LegacySector" in str(type(s)):
                s.mode = "Calibration"
                sectors.append(s)
                idx.append(i)

        getLogger(__name__).info("Calibrating LegacySectors...")

        if 2015 in self.time_framework:
            hist_years = self.time_framework[where(self.time_framework <= 2015)]
        hist = len(hist_years)
        for year_idx in range(hist):  # range(nyear):
            years = self.time_framework[year_idx : year_idx + 1]
            sectors = deepcopy(sectors)
            variables = ["supply", "consumption", "prices"]
            new_market = self.market[variables].sel(year=years).copy(deep=True)
            for sector in sectors:
                sector_market = sector.next(
                    new_market[["supply", "consumption", "prices"]]  # type:ignore
                )

                sector_market = sector_market.sel(year=new_market.year)

                dims = {i: sector_market[i] for i in sector_market.consumption.dims}

                sector_market.consumption.loc[dims] = clip(
                    sector_market.consumption.loc[dims]
                    - sector_market.supply.loc[dims],
                    0.0,
                    None,
                )
                new_market.consumption.loc[dims] += sector_market.consumption

                dims = {i: sector_market[i] for i in sector_market.supply.dims}
                new_market.supply.loc[dims] += sector_market.supply

        for i, s in enumerate(sectors):
            s.mode = "Iteration"
            self.sectors[idx[i]] = s

        getLogger(__name__).info("Finish calibration of LegacySectors!")

        return None, self.sectors, hist_years


class SingleYearIterationResult(NamedTuple):
    """Result of iterating over sectors for a year.

    Convenience tuple naming naming the return values from  of
    :py:func:`single_year_iteration`.
    """

    market: Dataset
    sectors: List[AbstractSector]


def single_year_iteration(
    market: Dataset, sectors: List[AbstractSector]
) -> SingleYearIterationResult:
    """Runs one iteration of the sectors (runs each sector once).

    Arguments:
        market: An initial market with prices, supply, consumption.
        sectors: A list of the sectors participating in the simulation.

    Returns:
        A tuple with the new market and sectors.
    """
    from copy import deepcopy

    from numpy import clip

    from muse.commodities import is_enduse

    sectors = deepcopy(sectors)
    market = market.copy(deep=True)
    if "updated_prices" not in market.data_vars:
        market["updated_prices"] = market.prices.copy()

    # eventually, the first market should be one that creates the initial demand
    for sector in sectors:
        sector_market = sector.next(
            market[["supply", "consumption", "prices"]]  # type:ignore
        )

        sector_market = sector_market.sel(year=market.year)

        dims = {i: sector_market[i] for i in sector_market.consumption.dims}

        sector_market.consumption.loc[dims] = clip(
            sector_market.consumption.loc[dims] - sector_market.supply.loc[dims],
            0.0,
            None,
        )

        market.consumption.loc[dims] += sector_market.consumption

        dims = {i: sector_market[i] for i in sector_market.supply.dims}
        market.supply.loc[dims] += sector_market.supply

        costs = sector_market.costs.sel(commodity=is_enduse(sector_market.comm_usage))

        # do not write costs lower than 1e-4
        # should correspond to rounding value
        if len(costs.commodity) > 0:
            costs = costs.where(costs > 1e-4, 0)
            dims = {i: costs[i] for i in costs.dims}
            costs = costs.where(costs > 0, market.prices.loc[dims])
            market.updated_prices.loc[dims] = costs.transpose(
                *market.updated_prices.dims
            )

    return SingleYearIterationResult(market, sectors)


class FindEquilibriumResults(NamedTuple):
    """Result of find equilibrium."""

    converged: bool
    market: Dataset
    sectors: List[AbstractSector]


def find_equilibrium(
    market: Dataset,
    sectors: List[AbstractSector],
    maxiter: int = 3,
    tol: float = 0.1,
    equilibrium_variable: Text = "demand",
    tol_unmet_demand: float = -0.1,
    excluded_commodities: Optional[Sequence] = None,
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

    from muse.utilities import future_propagation

    market = market.copy(deep=True)
    if excluded_commodities:
        included = ~market.commodity.isin(excluded_commodities)
    else:
        included = ones(len(market.commodity), dtype=bool)

    market["updated_prices"] = market.prices.copy()
    prior_market = market.copy(deep=True)
    converged = False
    equilibrium_sectors = sectors
    for i in range(maxiter):
        market["prices"] = market.updated_prices
        prior_market, market = market, prior_market
        market.consumption[:] = 0.0
        market.supply[:] = 0.0
        market, equilibrium_sectors = single_year_iteration(market, sectors)

        check_demand_fulfillment(market.sel(commodity=included), tol_unmet_demand)

        equilibrium_reached = check_equilibrium(
            market.sel(commodity=included),
            prior_market.sel(commodity=included),
            tol,
            equilibrium_variable,
            market.year[1],
        )

        if equilibrium_reached:
            converged = True
            new_price = prior_market["prices"].sel(year=market.year[1]).copy()
            new_price.loc[dict(commodity=included)] = market.updated_prices.sel(
                commodity=included, year=market.year[1]
            )
            market["prices"] = future_propagation(  # type: ignore
                market["prices"], new_price
            )

            break

        if equilibrium and not converged:
            new_price = prior_market["prices"].sel(year=market.year[1]).copy()
            new_price.loc[dict(commodity=included)] = (  # type: ignore
                0.8 * new_price.loc[dict(commodity=included)]  # type: ignore
            )
            new_price.loc[dict(commodity=included)] += (
                0.2
                * market.updated_prices.loc[
                    dict(year=market.year[1], commodity=included)
                ]
            )
            market["prices"] = future_propagation(  # type: ignore
                market["prices"], new_price
            )
        if not equilibrium:
            equilibrium_reached = True
            converged = True
            break

        if maxiter == 1:
            equilibrium_reached = True
            converged = True
            break

    else:
        # We didn't reached convergence and the loop is over: the simulation failed!
        msg = (
            f"CONVERGENCE ERROR: Maximum number of iterations ({maxiter}) reached "
            f"in year {int(market.year[0])}"
        )
        new_price = prior_market["prices"].sel(year=market.year[1]).copy()
        new_price.loc[dict(commodity=included)] = (
            0.8 * new_price.loc[dict(commodity=included)]  # type: ignore
        )
        new_price.loc[dict(commodity=included)] += (
            0.2
            * market.updated_prices.loc[dict(year=market.year[1], commodity=included)]
        )
        market["prices"] = future_propagation(  # type: ignore
            market["prices"], new_price
        )
        getLogger(__name__).critical(msg)

    return FindEquilibriumResults(
        converged, market.drop_vars("updated_prices"), equilibrium_sectors
    )


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

    future = market.year[-1]
    delta = (market.supply - market.consumption).sel(year=future)
    unmet = (delta < tol).any([u for u in delta.dims if u != "commodity"])

    if unmet.any():
        commodities = ", ".join(unmet.commodity.sel(commodity=unmet.values).values)
        getLogger(__name__).warning(f"Check growth constraints for {commodities}.")

        return False

    return True


def check_equilibrium(
    market: Dataset,
    int_market: Dataset,
    tolerance: float,
    equilibrium_variable: Text,
    year: Optional[int] = None,
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
