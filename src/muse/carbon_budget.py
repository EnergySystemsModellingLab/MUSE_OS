from collections.abc import MutableMapping, Sequence
from typing import Callable

import numpy as np
import xarray as xr
from scipy.optimize import curve_fit

from muse.mca import FindEquilibriumResults
from muse.registration import registrator

CARBON_BUDGET_METHODS_SIGNATURE = Callable[
    [xr.Dataset, Callable, xr.DataArray, list], float
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
    equilibrium: Callable[[xr.Dataset], FindEquilibriumResults],
    carbon_budget: xr.DataArray,
    commodities: list,
    refine_price: bool = False,
    price_too_high_threshold: float = 10,
    sample_size: int = 5,
    fitter: str = "linear",
    resolution: int = 2,
) -> float:
    """Used to solve the carbon market.

    Given the emission of a period, adjusts carbon price to meet the budget. A carbon
    market is meant as a pool of emissions for all the modelled regions; therefore, the
    carbon price applies to all modelled regions. The method solves an equation applying
    a fitting of the emission-carbon price relation.

    Arguments:
        market: Market, with the prices, supply, and consumption
        equilibrium: Method for searching market equilibrium
        carbon_budget: limit on emissions
        commodities: list of commodities to limit (ie. emissions)
        refine_price: Boolean to decide on whether carbon price should be capped, with
            the upper bound given by price_too_high_threshold
        price_too_high_threshold: threshold on carbon price
        sample_size: sample size for fitting
        fitter: method to fit emissions with carbon price
        resolution: Number of decimal places to solve the carbon price to

    Returns:
        new_price: adjusted carbon price to meet budget
    """
    # Calculate the carbon price and emissions threshold in the forecast year
    future = market.year[-1]
    threshold = carbon_budget.sel(year=future).values.item()
    price = market.prices.sel(year=future, commodity=commodities).mean().values.item()

    # Solve market with current carbon price
    emissions = solve_market(market, equilibrium, commodities, price)

    # Create a sample of prices at which we want to calculate emissions
    sample_prices = create_sample(price, emissions, threshold, sample_size)
    sample_emissions = np.zeros_like(sample_prices)
    sample_emissions[0] = emissions

    # For each sample price, we calculate the new emissions
    for i, new_price in enumerate(sample_prices[1:]):
        sample_emissions[i + 1] = solve_market(
            market, equilibrium, commodities, new_price
        )

    # Based on these results, we finally adjust the carbon price
    new_price = CARBON_BUDGET_FITTERS[fitter](
        sample_prices,
        sample_emissions,
        threshold,  # type: ignore
    )

    # Cap price between 0.0 and price_too_high_threshold
    if refine_price:
        new_price = min(new_price, price_too_high_threshold)
    new_price = max(new_price, 0.0)

    return round(new_price, resolution)


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
    equilibrium: Callable[[xr.Dataset], FindEquilibriumResults],
    carbon_budget: xr.DataArray,
    commodities: list,
    refine_price: bool = False,
    price_too_high_threshold: float = 10,
    max_iterations: int = 5,
    tolerance: float = 0.1,
    early_termination_count: int = 5,
    resolution: int = 2,
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
        equilibrium: Method for searching market equilibrium
        carbon_budget: DataArray with the carbon budget
        commodities: List of carbon-related commodities
        refine_price: Boolean to decide on whether carbon price should be capped, with
            the upper bound given by price_too_high_threshold
        price_too_high_threshold: Upper limit for carbon price
        max_iterations: Maximum number of iterations for bisection
        tolerance: Maximum permitted deviation of emissions from the budget
        early_termination_count: Will terminate the loop early if the last n solutions
            are the same
        resolution: Number of decimal places to solve the carbon price to

    Returns:
        New value of global carbon price
    """
    from logging import getLogger

    # Create cache for emissions at different price points
    emissions_cache = EmissionsCache(market, equilibrium, commodities)

    # Carbon price and emissions threshold in the forecast year
    future = market.year[-1]
    target = carbon_budget.sel(year=future).values.item()
    price = market.prices.sel(year=future, commodity=commodities).mean().values.item()

    # Test if emissions are already below the budget without imposing a carbon price
    if emissions_cache[0.0] < target:
        message = (
            f"Emissions for the year {int(future)} are already below the carbon budget "
            "without imposing a carbon price. The carbon price has been set to zero."
        )
        getLogger(__name__).warning(message)
        return 0.0

    # Initial lower and upper bounds on carbon price for the bisection algorithm
    current = market.year[0]
    time_exp = int(future - current)
    lb_price = 0.0
    epsilon = 10**-resolution  # smallest nonzero price
    ub_price = round(
        (max(price, epsilon) * 1.1**time_exp), resolution
    )  # i.e. 10% yearly increase on current price

    # Bisection loop
    for _ in range(max_iterations):  # maximum number of iterations before terminating
        # Cap prices between 0.0 and price_too_high_threshold
        if refine_price:
            ub_price = min(ub_price, price_too_high_threshold)
        lb_price = max(lb_price, 0.0)

        # Calculate/retrieve carbon emissions at new bounds
        lb_price_emissions = emissions_cache[lb_price]
        ub_price_emissions = emissions_cache[ub_price]

        # Terminate early if many consecutive emissions are the same
        if len(emissions_cache) >= early_termination_count + 2:
            recent_values = list(emissions_cache.values())[-early_termination_count:]
            if all(x == recent_values[0] for x in recent_values):
                break

        # Exit loop if lower or upper bound on emissions is close to threshold
        if abs(lb_price_emissions - target) <= abs(tolerance * target):
            return lb_price
        if abs(ub_price_emissions - target) <= abs(tolerance * target):
            return ub_price

        # Convergence not yet reached -> calculate new bounds
        lb_price, ub_price = adjust_bounds(
            lb_price,
            ub_price,
            emissions_cache,
            target,
            resolution,
        )

    # If convergence isn't reached, new price is that with emissions closest to
    # threshold. If multiple prices are equally close, it returns the lowest price
    new_price = min(
        emissions_cache, key=lambda k: (abs(emissions_cache[k] - target), k)
    )

    # Raise warning message
    if all(emissions_cache[k] > target for k in emissions_cache):
        message = (
            f"Carbon budget could not be met for the year {int(future)} "
            f"(budget: {target}, emissions: {emissions_cache[new_price]}). "
            "This may be because there are no processes available that can meet the "
            "budget, or because emissions from capacity installed earlier in the time "
            "horizon is preventing the budget from being met. "
            "The CO2 price in this year should be interpreted with caution."
        )
    else:
        message = (
            f"Carbon budget could not be matched for the year {int(future)} to within "
            "the specified tolerance. "
            "This is sometimes unavoidable due to a discontinuous emissions landscape "
            "which can make the budget unreachable, but can sometimes be "
            "fixed by increasing max_iterations, early_termination_count or resolution."
        )
    getLogger(__name__).warning(message)
    return new_price


class EmissionsCache(dict):
    """Cache of emissions at different price points for bisection algorithm.

    If a price is queried that is not in the cache, it calculates the emissions at that
    price using solve_market and stores the result in the cache.
    """

    def __init__(self, market, equilibrium, commodities):
        super().__init__()
        self.market = market
        self.equilibrium = equilibrium
        self.commodities = commodities

    def __missing__(self, price):
        value = solve_market(self.market, self.equilibrium, self.commodities, price)
        self[price] = value
        return value


def adjust_bounds(
    lb_price: float,
    ub_price: float,
    emissions_cache: dict[float, float],
    target: float,
    resolution: int = 2,
) -> tuple[float, float]:
    """Adjust the bounds of the carbon price for the bisection algorithm.

    As emissions can be a discontinuous function of the carbon price, this method is
    used to improve the solution search when discontinuities are met, improving the
    bounds search.

    Arguments:
        lb_price: Value of carbon price at lower bound
        ub_price: Value of carbon price at upper bound
        emissions_cache: Dictionary of emissions at different price points
        target: Carbon budget
        resolution: Number of decimal places to solve the carbon price to

    Returns:
        New lower and upper bounds for the carbon price.
    """
    lb_price_emissions = emissions_cache[lb_price]
    ub_price_emissions = emissions_cache[ub_price]

    if lb_price_emissions < target and ub_price_emissions < target:
        # Emissions too low at both prices -> decrease prices
        method = decrease_bounds

    elif lb_price_emissions > target and ub_price_emissions > target:
        # Emissions too high at both prices -> increase prices
        method = increase_bounds

    elif lb_price_emissions > target and ub_price_emissions < target:
        # Threshold is between bounds -> perform bisection
        method = bisect_bounds

    elif lb_price_emissions < target and ub_price_emissions > target:
        # Inverted bounds (i.e. increasing price leads to increasing emissions)
        # Unlikely case, but included for completeness
        method = bisect_bounds_inverted

    else:
        # lb_price_emissions or ub_price_emissions is equal to the target, so we can
        # return the bounds unchanged
        return lb_price, ub_price

    return method(
        lb_price,
        ub_price,
        emissions_cache,
        target,
        resolution,
    )


def decrease_bounds(
    lb_price: float,
    ub_price: float,
    emissions_cache: dict[float, float],
    target: float,
    resolution: int = 2,
) -> tuple[float, float]:
    """Decreases the lb of the carbon price, and sets the ub to the previous lb."""
    denominator = max(target, 1e-3)
    lb_price_emissions = emissions_cache[lb_price]
    exponent = (lb_price_emissions - target) / abs(denominator)  # will be negative
    exponent = max(exponent, -1)  # cap exponent at -1
    ub_price = lb_price
    lb_price = round(lb_price * np.exp(exponent), resolution)
    return lb_price, ub_price


def increase_bounds(
    lb_price: float,
    ub_price: float,
    emissions_cache: dict[float, float],
    target: float,
    resolution: int = 2,
) -> tuple[float, float]:
    """Increases the ub of the carbon price, and sets the lb to the previous ub."""
    denominator = max(target, 1e-3)
    ub_price_emissions = emissions_cache[ub_price]
    exponent = (ub_price_emissions - target) / abs(denominator)  # will be positive
    exponent = min(exponent, 1)  # cap exponent at 1
    lb_price = ub_price
    ub_price = round(ub_price * np.exp(exponent), resolution)
    return lb_price, ub_price


def bisect_bounds(
    lb_price: float,
    ub_price: float,
    emissions_cache: dict[float, float],
    target: float,
    resolution: int = 2,
) -> tuple[float, float]:
    """Bisects the bounds of the carbon price."""
    midpoint = round((lb_price + ub_price) / 2.0, resolution)
    midpoint_emissions = emissions_cache[midpoint]
    if midpoint_emissions < target:
        ub_price = midpoint
    else:
        lb_price = midpoint
    return lb_price, ub_price


def bisect_bounds_inverted(
    lb_price: float,
    ub_price: float,
    emissions_cache: dict[float, float],
    target: float,
    resolution: int = 2,
) -> tuple[float, float]:
    """Bisects the bounds of the carbon price, in the case of inverted bounds."""
    midpoint = round((lb_price + ub_price) / 2.0, resolution)
    midpoint_emissions = emissions_cache[midpoint]
    if midpoint_emissions > target:
        ub_price = midpoint
    else:
        lb_price = midpoint
    return lb_price, ub_price


def solve_market(
    market: xr.Dataset,
    equilibrium: Callable[[xr.Dataset], FindEquilibriumResults],
    commodities: list,
    carbon_price: float,
) -> float:
    """Solves the market with a new carbon price and returns the emissions.

    Arguments:
        market: Market, with the prices, supply, consumption and demand
        equilibrium: Method for searching market equilibrium
        commodities: List of carbon-related commodities
        carbon_price: New carbon price

    Returns:
        Emissions at the new carbon price.
    """
    future = market.year[-1]
    new_market = market.copy(deep=True)

    # Assign new carbon price and solve market
    new_market.prices.loc[{"year": future, "commodity": commodities}] = carbon_price
    new_market = equilibrium(new_market).market
    new_emissions = (
        new_market.supply.sel(year=future, commodity=commodities)
        .sum(["region", "timeslice", "commodity"])
        .round(decimals=2)
    ).values.item()

    return new_emissions
