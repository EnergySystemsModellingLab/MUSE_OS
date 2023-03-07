from typing import Callable, MutableMapping, Sequence, Text

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


CARBON_BUDGET_METHODS: MutableMapping[Text, CARBON_BUDGET_METHODS_SIGNATURE] = {}
"""Dictionary of carbon budget methods checks."""

CARBON_BUDGET_FITTERS: MutableMapping[Text, CARBON_BUDGET_FITTERS_SIGNATURE] = {}
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

    Returns:
        An adjusted threshold for the far future year
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
    fitter: Text = "slinear",
) -> float:
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
        sample_prices, sample_emissions, threshold  # type: ignore
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
    """Refine the value of the carbon price to ensure it is not too high or low.
    Arguments:
        market: Market, with the prices, supply, consumption and demand.
        historic_price: DataArray with the historic carbon prices.
        carbon_budget: DataArray with the carbon budget.
        sample: Sample carbon price points.
        price: Current carbon price, to be refined.
        commodities: List of carbon-related commodities.
        price_too_high_threshold: Threshold to decide what is a price too high.

    Returns:
        A refined carbon price.
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
    """Calculates a sample of carbon prices to estimate the adjusted carbon
    price.

    For each of these prices, the equilibrium loop will be run, obtaining a new value
    for the emissions. Out of those price-emissions pairs, the final carbon price will
    be estimated.

    Arguments:
        carbon_price: Current carbon price
        current_emissions: Current emissions
        budget: Carbon budget
        size: Number of points in the sample

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
        prices: An array with the sample carbon prices
        emissions: An array with the corresponding emissions
        budget: The carbon budget for the time period

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
    """Estimates initial values for the linear fitting algorithm and the
    weights.

    The points closest to the budget are used to estimate the initial guess. They also
    have the highest weight.

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
        prices: An array with the sample carbon prices
        emissions: An array with the corresponding emissions
        budget: The carbon budget for the time period

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
    """Estimates initial values for the exponential fitting algorithm and the
    weights.

    The points closest to the budget are used to estimate the initial guess. They also
    have the highest weight.

    Arguments:
        prices: An array with the sample carbon prices
        emissions: An array with the corresponding emissions
        budget: The carbon budget for the time period

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
    fitter: Text = "slinear",
) -> float:
    # to be set as moving value in superloop before emissions start increasing
    current = carbon_price.year.min() + sectors[-1].forecast
    future = market.year[-1]
    threshold = carbon_budget.sel(year=future).values
    price = market.prices.sel(year=future, commodity=commodities).mean().values
    print(carbon_price, "carbon price")
    niter = sample_size
    # We create a sample of prices at which we want to calculate emissions
    time_exp = max(0, (int(future - current)))
    small = round((1 + 0.01) ** time_exp, 4)
    large = round((1 + 0.02) ** time_exp, 4)

    sample_prices = price * np.linspace(small, large, 2, endpoint=True)

    low0 = round(min(sample_prices), 7)
    up0 = round(max(sample_prices), 7)
    lb = bisect_loop(market, sectors, equilibrium, commodities, up0)
    ub = bisect_loop(market, sectors, equilibrium, commodities, low0)
    for n in range(niter):
        print(n, "iteration bisection")
        if refine_price:
            if max(sample_prices) > price_too_high_threshold:
                price_too_high_threshold = round(
                    price_too_high_threshold * (1 + 0.1) ** time_exp, 7
                )
                up0 = round(min(max(sample_prices), price_too_high_threshold), 7)
                low0 = round(
                    min(min(sample_prices), price_too_high_threshold, up0 * 0.9), 7
                )
                lb = bisect_loop(market, sectors, equilibrium, commodities, up0)
                ub = bisect_loop(market, sectors, equilibrium, commodities, low0)
        if low0 == up0:
            new_price = low0
            break
        if lb == threshold:
            new_price = up0
            break
        elif ub == threshold:
            new_price = low0
            break
        else:
            low, up = min_max_bisect(
                low0,
                lb,
                up0,
                ub,
                market,
                sectors,
                equilibrium,
                commodities,
                threshold,  # type: ignore
            )
            if abs(low - up) <= 0.001:
                new_price = round((low + up) / 2.0, 7)
                print("5")
                break
            elif abs(ub - threshold) <= abs(0.1 * threshold):
                new_price = low
                print("6")
                break
            elif abs(lb - threshold) <= abs(0.1 * threshold):
                new_price = up
                print("7")
                break
            if low != low0:
                low0 = low
                ub = bisect_loop(market, sectors, equilibrium, commodities, low)
            if up != up0:
                up0 = up
                lb = bisect_loop(market, sectors, equilibrium, commodities, up)
            new_price = round((low + up) / 2.0, 7)
            print(low0, low, up0, up, "8")

    if new_price <= 0:
        new_price = 1e-2
    print(new_price, low0, up0, "new_price")
    return new_price


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
):
    denominator = threshold if threshold != 0.0 else 1e-3
    if lb < threshold and ub < threshold:
        # ub too small -> decrease low
        # negative exponent (-(threshold - ub)) < 1
        pow = -(threshold - lb) / abs(denominator)
        pow = pow if pow < 1 else 1
        pow = pow if pow > -1 else -1
        up = low if ub > lb else up
        low = low * np.exp(pow)
        low = low if low > 0.0 else 1e-3
        print("1")

    if ub > threshold and lb > threshold:
        # lb too big -> increase up
        # positive exponent (-(threshold - lb)) < 1
        pow = -2 * (threshold - lb) / abs(denominator)
        pow = pow if pow < 1 else 1
        pow = pow if pow > -2 else -2
        low = up if lb < ub else low
        up = up * np.exp(pow)
        up = up if up > 0.0 else 1e-3
        print("2")

    if ub > threshold and lb < threshold:
        midpoint = round((low + up) / 2.0, 7)
        m = bisect_loop(market, sectors, equilibrium, commodities, midpoint)
        if m < threshold:
            # midpoint is smaller than lb
            # lb is a function of up price
            up = midpoint
            print("3a")
        else:
            # midpoint is smaller than ub
            # ub is a function of low price
            low = midpoint
            print("4a")

    if ub < threshold and lb > threshold:
        "Inverted bounds"
        low1 = up
        up = low
        low = low1
        if m < threshold:
            # midpoint is smaller than lb
            # lb is a function of up price
            up = midpoint
            print("3b")
        else:
            # midpoint is smaller than ub
            # ub is a function of low price
            low = midpoint
            print("4b")

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
    future = market.year[-1]
    new_market = market.copy(deep=True)
    # Assign new carbon price
    new_market.prices.loc[{"year": future, "commodity": commodities}] = new_price

    new_market = equilibrium(new_market, sectors, 1).market

    new_emissions = (
        new_market.supply.sel(year=future, commodity=commodities)
        .sum(["region", "timeslice", "commodity"])
        .round(decimals=3)
    ).values

    return new_emissions
