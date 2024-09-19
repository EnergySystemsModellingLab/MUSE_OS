from collections.abc import MutableMapping, Sequence
from typing import Callable

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from muse.mca import FindEquilibriumResults
from muse.registration import registrator
from muse.sectors import AbstractSector

CARBON_BUDGET_METHODS_SIGNATURE = Callable[
    [xr.Dataset, list, Callable, xr.DataArray, xr.DataArray], float
]
"""carbon budget fitters signature."""

CARBON_BUDGET_FITTERS_SIGNATURE = Callable[[np.ndarray, np.ndarray, int], float]
"""carbon budget fitters signature."""


CARBON_BUDGET_METHODS: MutableMapping[str, CARBON_BUDGET_METHODS_SIGNATURE] = {}
"""Dictionary of carbon budget methods checks."""

CARBON_BUDGET_FITTERS: MutableMapping[str, CARBON_BUDGET_FITTERS_SIGNATURE] = {}
"""Dictionary of carbon budget fitters."""


@registrator(registry=CARBON_BUDGET_FITTERS, loglevel="debug")
def register_carbon_budget_fitter(function: CARBON_BUDGET_FITTERS_SIGNATURE = None):
    """Decorator to register a carbon budget function."""
    return function


@registrator(registry=CARBON_BUDGET_METHODS, loglevel="debug")
def register_carbon_budget_method(function: CARBON_BUDGET_METHODS_SIGNATURE = None):
    """Decorator to register a carbon budget function."""
    return function


def update_carbon_budget(
    carbon_budget: Sequence[float],
    emissions: float,
    year_idx: int,
    over: bool = True,
    under: bool = True,
) -> float:
    """Adjust the carbon budget in the far future if emissions too high or low.

    This feature can allow to simulate overshoot shifting.

    Arguments:
        carbon_budget: budget for future year,
        emissions: emission for future year,
        year_idx: index of year for estimation,
        over: if True, allows overshoot,
        under: if True, allows undershoot.

    Returns:
        An adjusted threshold for the future year
    """
    delta = emissions - carbon_budget[year_idx + 1]
    new_future_budget = carbon_budget[year_idx + 2]
    if over:
        new_future_budget -= max(delta, 0)
    if under:
        new_future_budget -= min(delta, 0)

    return new_future_budget


@register_carbon_budget_method
def fitting(
    market: xr.Dataset,
    sectors: list,
    equilibrium: Callable[
        [xr.Dataset, Sequence[AbstractSector]], FindEquilibriumResults
    ],
    carbon_budget: xr.DataArray,
    carbon_price: xr.DataArray,
    commodities: list,
    sample_size: int = 4,
    refine_price: bool = True,
    price_too_high_threshold: float = 10,
    fitter: str = "slinear",
) -> float:
    """Used to solve the carbon market.

    Given the emission of a period, adjusts carbon price to meet the budget. A carbon
    market is meant as a pool of emissions for all the modelled regions; therefore, the
    carbon price applies to all modelled regions. The method solves an equation applying
    a fitting of the emission-carbon price relation.

    Arguments:
        market: Market, with the prices, supply, and consumption,
        sectors: list of market sectors,
        equilibrium: Method for searching market equilibrium,
        carbon_budget: limit on emissions,
        carbon_price: current carbon price
        commodities: list of commodities to limit (ie. emissions),
        sample_size: sample size for fitting,
        refine_price: if True, performs checks on estimated carbon price,
        price_too_high_threshold: threshold on carbon price,
        fitter: method to fit emissions with carbon price.

    Returns:
        new_price: adjusted carbon price to meet budget
    """
    future = market.year[-1]

    threshold = carbon_budget.sel(year=future).values
    emissions = market.supply.sel(year=future, commodity=commodities).sum().values
    price = market.prices.sel(year=future, commodity=commodities).mean().values

    # We create a sample of prices at which we want to calculate emissions
    sample_prices = create_sample(price, emissions, threshold, sample_size)
    sample_emissions = np.zeros_like(sample_prices)
    sample_emissions[0] = emissions

    # For each sample price, we calculate the new emissions
    new_market = None
    for i, new_price in enumerate(sample_prices[1:]):
        # Reset market and sectors
        new_market = market.copy(deep=True)

        # Assign new carbon price
        new_market.prices.loc[{"year": future, "commodity": commodities}] = new_price

        new_market = equilibrium(new_market, sectors).market

        sample_emissions[i + 1] = new_market.supply.sel(
            year=future, commodity=commodities
        ).sum(["region", "timeslice", "commodity"])

    # Based on these results, we finally adjust the carbon price
    new_price = CARBON_BUDGET_FITTERS[fitter](
        sample_prices,
        sample_emissions,
        threshold,  # type: ignore
    )

    if refine_price and new_market is not None:
        new_price = refine_new_price(
            new_market,
            carbon_price,
            carbon_budget,
            sample_prices,
            new_price,
            commodities,
            price_too_high_threshold,
        )

    return new_price


def refine_new_price(
    market: xr.Dataset,
    historic_price: xr.DataArray,
    carbon_budget: xr.DataArray,
    sample: np.ndarray,
    price: float,
    commodities: list,
    price_too_high_threshold: float,
) -> float:
    """Refine the value of the carbon price.

    Ensure it is not too high or low compared to heuristic values.

    Arguments:
        market: Market, with prices, supply, and consumption,
        historic_price: DataArray with the historic carbon prices,
        carbon_budget: DataArray with the carbon budget,
        sample: Sample carbon price points,
        price: Current carbon price, to be refined,
        commodities: List of carbon-related commodities,
        price_too_high_threshold: Threshold to decide what is a price too high.

    Returns:
        The new carbon price
    """
    future = market.year[-1]

    emissions = (
        market.supply.sel(year=future, commodity=commodities)
        .sum(["region", "timeslice", "commodity"])
        .values
    )

    carbon_price = historic_price.sel(year=historic_price.year < future).values

    if (carbon_price[-2:] > 0).all():
        relative_price_increase = np.diff(carbon_price) / carbon_price[-1]
        average = np.mean(relative_price_increase)
    else:
        average = 0.2

    if price > price_too_high_threshold:  # * max(min(carbon_price), 0.1):
        price = min(price_too_high_threshold, max(sample) * (1 + average))
    elif price <= 0:
        threshold = carbon_budget.sel(year=future).values
        exponent = (emissions - threshold) / threshold
        magnitude = max(1 - np.exp(exponent), -0.1)
        price = min(sample) * (1 + magnitude)

    return price


def linear_fun(x, a, b):
    return a + b * x


def exponential_fun(x, a, b, c):
    return a * np.exp(-b * x) + c


def create_sample(carbon_price, current_emissions, budget, size=4):
    """Calculates a sample of carbon prices to estimate the adjusted carbon price.

    For each of these prices, the equilibrium loop will be run, obtaining a new value
    for the emissions. Out of those price-emissions pairs, the final carbon price will
    be estimated.

    Arguments:
        carbon_price: Current carbon price,
        current_emissions: Current emissions,
        budget: Carbon budget,
        size: Number of points in the sample.

    Returns:
        An array with the sample prices.
    """
    exponent = (current_emissions - budget) / budget
    magnitude = max(1 - np.exp(-exponent), -0.1)

    sample = carbon_price * (1 + np.linspace(0, 1, size) * magnitude)

    return np.abs(sample)


@register_carbon_budget_fitter
def linear(prices: np.ndarray, emissions: np.ndarray, budget: int) -> float:
    """Fits the prices-emissions pairs to a linear function.

    Once that is done, an optimal carbon price is estimated

    Arguments:
        prices: An array with the sample carbon prices,
        emissions: An array with the corresponding emissions,
        budget: The carbon budget for the time period.

    Returns:
        The optimal carbon price.
    """
    guess, weights = linear_guess_and_weights(prices, emissions, budget)

    sol, _ = curve_fit(
        linear_fun, emissions, prices, guess, sigma=weights, absolute_sigma=True
    )

    new_price = linear_fun(budget, *sol)

    return new_price


def linear_guess_and_weights(
    prices: np.ndarray, emissions: np.ndarray, budget: int
) -> tuple:
    """Estimates initial values for the linear fitting algorithm and the weights.

    The points closest to the budget are used to estimate the initial guess. They also
    have the highest weight.

    Arguments:
        prices: An array with the sample carbon prices,
        emissions: An array with the corresponding emissions,
        budget: The carbon budget for the time period.

    Returns:
        The initial guess and weights
    """
    weights = np.abs(emissions - budget)

    idx = np.argsort(weights)

    p = prices[idx][:2]
    e = emissions[idx][:2]

    den = e[1] - e[0]
    num = p[1] - p[0]

    if den != 0:
        b = num / den
        a = p[0] - b * e[0]
    else:
        b = 0
        a = p[0]

    return (a, b), weights


@register_carbon_budget_fitter
def exponential(prices: np.ndarray, emissions: np.ndarray, budget: int) -> float:
    """Fits the prices-emissions pairs to an exponential function.

    Once that is done, an optimal carbon price is estimated

    Arguments:
        prices: An array with the sample carbon prices,
        emissions: An array with the corresponding emissions,
        budget: The carbon budget for the time period.

    Returns:
        The optimal carbon price.
    """
    guess, weights = exp_guess_and_weights(prices, emissions, budget)

    sol, _ = curve_fit(
        exponential_fun, emissions, prices, guess, sigma=weights, absolute_sigma=True
    )

    new_price = exponential_fun(budget, *sol)

    return new_price


def exp_guess_and_weights(
    prices: np.ndarray, emissions: np.ndarray, budget: int
) -> tuple:
    """Estimates initial values for the exponential fitting algorithm and the weights.

    The points closest to the budget are used to estimate the initial guess. They also
    have the highest weight.

    Arguments:
        prices: An array with the sample carbon prices,
        emissions: An array with the corresponding emissions,
        budget: The carbon budget for the time period.

    Returns:
        The initial guess and weights
    """
    weights = np.abs(emissions - budget)

    idx = np.argsort(weights)

    p = prices[idx][:2]
    e = emissions[idx][:2]

    b = 1 / e[0]
    e = np.exp(-b * e)

    den = e[1] - e[0]
    num = p[1] - p[0]

    if den != 0:
        a = num / den
        c = p[0] - a * e[0]
    else:
        a = 0
        c = p[0]
    return (a, b, c), weights


@register_carbon_budget_method
def bisection(
    market: xr.Dataset,
    sectors: list,
    equilibrium: Callable[
        [xr.Dataset, Sequence[AbstractSector], int], FindEquilibriumResults
    ],
    carbon_budget: xr.DataArray,
    carbon_price: xr.DataArray,
    commodities: list,
    sample_size: int = 2,
    refine_price: bool = True,
    price_too_high_threshold: float = 10,
    fitter: str = "slinear",
    tolerance: float = 0.1,
) -> float:
    """Applies bisection algorithm to escalate carbon price and meet the budget.

    A carbon market is meant as a pool of emissions for all
    the modelled regions; therefore, the carbon price applies to all modelled regions.
    Bisection applies an iterative estimations of the emissions
    varying the carbon price until convergence or stop criteria
    are met.
    Builds on 'register_carbon_budget_method'.

    Arguments:
        market: Market, with the prices, supply, consumption and demand
        sectors: List of sectors
        equilibrium: Method for searching market equilibrium
        carbon_budget: DataArray with the carbon budget
        carbon_price: DataArray with the carbon price
        commodities: List of carbon-related commodities
        sample_size: Maximum number of iterations for bisection
        refine_price: Boolean to decide on whether carbon price should be capped, with
            the upper bound given by price_too_high_threshold
        price_too_high_threshold: Upper limit for carbon price
        fitter: Not used in this method
        tolerance: Maximum permitted deviation of emissions from the budget

    Returns:
        New value of global carbon price
    """
    from logging import getLogger

    # We calculate the carbon price and emissions threshold in the forecast year
    future = market.year[-1]
    threshold = carbon_budget.sel(year=future).values.item()
    price = market.prices.sel(year=future, commodity=commodities).mean().values.item()

    # Initial lower and upper bounds on carbon price for the bisection algorithm
    current = market.year[0]
    time_exp = int(future - current)
    low = 0.01
    up = max(price, 1e-2) * 1.1**time_exp  # i.e. 10% yearly increase on current price

    # Bisection loop
    emissions_cache: dict[float, float] = {}  # caches emissions for different prices
    for _ in range(sample_size):  # maximum number of iterations before terminating
        # Cap prices between 0.01 and price_too_high_threshold
        if refine_price:
            up = min(up, price_too_high_threshold)
        low = max(low, 0.01)

        # Round prices to 3dp
        low = round(low, 3)
        up = round(up, 3)

        # Calculate carbon emissions at new bounds
        if low not in emissions_cache:
            emissions_cache[low] = bisect_loop(
                market, sectors, equilibrium, commodities, low
            )
        ub = emissions_cache[low]
        if up not in emissions_cache:
            emissions_cache[up] = bisect_loop(
                market, sectors, equilibrium, commodities, up
            )
        lb = emissions_cache[up]

        # Terminate early if the last 5 solutions are the same
        if len(emissions_cache) >= 7:
            last_five = list(emissions_cache.values())[-5:]
            if all(x == last_five[0] for x in last_five):
                break

        # Exit loop if lower or upper bound on emissions is close to threshold
        if abs(ub - threshold) <= abs(tolerance * threshold):
            return low
        if abs(lb - threshold) <= abs(tolerance * threshold):
            return up

        # Convergence not yet reached -> calculate new bounds
        low, up = min_max_bisect(
            low,
            lb,
            up,
            ub,
            market,
            sectors,
            equilibrium,
            commodities,
            threshold,  # type: ignore
            emissions_cache=emissions_cache,
        )

    # If convergence isn't reached, new price is that with emissions closest to
    # threshold. If multiple prices are equally close, it returns the lowest price
    message = (
        f"Carbon budget could not be matched for year {int(future)}. "
        "Try increasing the tolerance or sample size."
    )
    getLogger(__name__).warning(message)
    return min(emissions_cache, key=lambda k: (abs(emissions_cache[k] - threshold), k))


def min_max_bisect(
    low: float,
    lb: float,
    up: float,
    ub: float,
    market: xr.Dataset,
    sectors: list,
    equilibrium: Callable[
        [xr.Dataset, Sequence[AbstractSector], int], FindEquilibriumResults
    ],
    commodities: list,
    threshold: float,
    emissions_cache: dict[float, float] = {},
):
    """Refines bisection algorithm to escalate carbon price and meet the budget.

    As emissions can be a discontinuous function of the carbon price, this method is
    used to improve the solution search when discontinuities are met, improving the
    bounds search.

    Arguments:
        low: Value of carbon price at lower bound
        lb: Value of emissions at lower bound
        up: Value of carbon price at upper bound
        ub: Value of emissions at upper bound
        market: Market, with the prices, supply, consumption and demand
        sectors: List of sectors
        equilibrium: Method for searching market equilibrium
        commodities: List of carbon-related commodities
        threshold: Carbon budget
        emissions_cache: Cache of emissions for carbon prices

    Returns:
        Value of lower and upper global carbon price
    """
    denominator = max(threshold, 1e-3)
    if lb < threshold and ub < threshold:
        # Both prices are too high (emissions too low) -> decrease the lower bound
        exp = (lb - threshold) / abs(denominator)  # will be negative
        exp = max(exp, -1)  # cap exponent at -1
        up = low if (ub >= lb) else up  # new upper bound is price with higher emissions
        low = low * np.exp(exp)

    if ub > threshold and lb > threshold:
        # Both prices are too low (emissions too high) -> increase the upper bound
        exp = 2 * (lb - threshold) / abs(denominator)  # will be positive
        exp = min(exp, 1)  # cap exponent at 1
        low = up if (ub >= lb) else low  # new lower bound is price with lower emissions
        up = up * np.exp(exp)

    if ub > threshold and lb < threshold:
        # Threshold is between bounds -> perform bisection
        midpoint = (low + up) / 2.0
        m = bisect_loop(market, sectors, equilibrium, commodities, midpoint)
        emissions_cache[midpoint] = m
        if m < threshold:
            # Midpoint price is too high -> becomes new upper bound
            up = midpoint
        else:
            # Midpoint price is too low -> becomes new lower bound
            low = midpoint

    # Inverted bounds (i.e. increasing price leads to increasing emissions)
    # Unlikely case, but included for completeness
    if ub < threshold and lb > threshold:
        midpoint = (low + up) / 2.0
        m = bisect_loop(market, sectors, equilibrium, commodities, midpoint)
        emissions_cache[midpoint] = m
        if m > threshold:
            # Midpoint price is too high -> becomes new upper bound
            up = midpoint
        else:
            # Midpoint price is too low -> becomes new lower bound
            low = midpoint

    return low, up


def bisect_loop(
    market: xr.Dataset,
    sectors: list,
    equilibrium: Callable[
        [xr.Dataset, Sequence[AbstractSector], int], FindEquilibriumResults
    ],
    commodities: list,
    new_price: float,
) -> float:
    """Calls market equilibrium iteration in bisection.

    This updates emissions during iterations.

    Arguments:
        market: Market, with the prices, supply, consumption and demand,
        sectors: List of sectors,
        equilibrium: Method for searching market equilibrium,
        commodities: List of carbon-related commodities,
        new_price: New carbon price from bisection,

    Returns:
        Emissions estimated at the new carbon price.
    """
    future = market.year[-1]
    new_market = market.copy(deep=True)

    # Assign new carbon price and solve market
    new_market.prices.loc[{"year": future, "commodity": commodities}] = new_price
    new_market = equilibrium(new_market, sectors).market
    new_emissions = (
        new_market.supply.sel(year=future, commodity=commodities)
        .sum(["region", "timeslice", "commodity"])
        .round(decimals=3)
    ).values.item()

    return new_emissions
