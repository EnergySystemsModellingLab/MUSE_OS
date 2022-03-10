import numpy as np
import xarray as xr
from typing import Callable, Text, Sequence
from muse.mca import FindEquilibriumResults
from muse.sectors import AbstractSector
from muse.carbon_budget import register_carbon_budget_method, refine_new_price
from muse.agents import Agent
from muse.filters import register_filter


@register_filter
def ec_budget(
    agent: Agent,
    search_space: xr.DataArray,
    technologies: xr.Dataset,
    market: xr.Dataset,
    enduse_label: Text = "service",
    **kwargs
) -> xr.DataArray:
    """Only allows technologies that have achieve a given market share.

    Specifically, the market share refers to the capacity for each end- use.
    """
    from muse.commodities import is_enduse

    # print (agent, 'agent')
    # budget.
    # unit_capex = agent.filter_input(
    #     technologies.cap_par
    # )

    return search_space & (unit_capex <= 5).all("commodity")


@register_carbon_budget_method
def bisection_plug(
    market: xr.Dataset,
    sectors: list,
    equi: Callable[[xr.Dataset, Sequence[AbstractSector]], FindEquilibriumResults],
    carbon_budget: xr.DataArray,
    carbon_price: xr.DataArray,
    commodities: list,
    sample_size: int = 10,
    refine_price: bool = True,
    price_too_high_threshold: float = 10,
    fitter: Text = "slinear",
) -> float:
    future = market.year[-1]
    # current = market.year[0]
    thr = carbon_budget.sel(year=future).values
    price = market.prices.sel(year=future, commodity=commodities).mean().values
    niter = sample_size
    # We create a sample of prices at which we want to calculate emissions
    sample_prices = price * np.linspace(0.9, 3, 2, endpoint=True)
    print("=======")
    print(price)
    print(sample_prices)
    print("sample prices")
    if refine_new_price:
        if max(sample_prices) > price_too_high_threshold:
            price_too_high_threshold = round(
                price_too_high_threshold * (1 + 0.01) ** (int(future - 2020)), 7
            )
        up = round(min(max(sample_prices), price_too_high_threshold), 7)
        low = round(min(min(sample_prices), price_too_high_threshold, up * 0.9), 7)
        lb = blp(market, sectors, equi, commodities, low) - thr
        ub = blp(market, sectors, equi, commodities, up) - thr
        new_price = mmb(niter, low, lb, up, ub, market, sectors, equi, commodities, thr)
    else:
        low = round(min(sample_prices), 7)
        up = round(max(sample_prices), 7)
        lb = blp(market, sectors, equi, commodities, low) - thr
        ub = blp(market, sectors, equi, commodities, up) - thr
        new_price = mmb(niter, low, lb, up, ub, market, sectors, equi, commodities, thr)

    return new_price


def mmb(
    niter: int,
    low: float,
    lb: float,
    up: float,
    ub: float,
    market: xr.Dataset,
    sectors: list,
    equi: Callable[[xr.Dataset, Sequence[AbstractSector]], FindEquilibriumResults],
    commodities: list,
    thr: float,
):
    future = market.year[-1]
    current = market.year[0]
    new_price = (up + low) / 2.0

    for n in range(niter):
        print("iteration n", n, " for year", current)
        if lb != ub:
            if lb * ub < 0:

                midpoint = round((low + up) / 2.0, 7)

                m = blp(market, sectors, equi, commodities, midpoint) - thr
                # Reset market and sectors

                if lb * m < 0:
                    low = midpoint
                else:
                    up = midpoint
                if (low - up) < 0.0001:
                    new_price = midpoint
                    break

            elif (
                lb > 0.0 and ub > 0.0
            ):  # covers also l==u: we are higher than emission limits
                up = round(up * (1 + 0.01) ** (int(future - current)), 7)
                ub = blp(market, sectors, equi, commodities, up) - thr
                new_price = up

            elif lb < 0.0 and ub < 0.0:  # lbis closer to the thr
                low = low / 2.0
                lb = blp(market, sectors, equi, commodities, low) - thr
                new_price = low

        else:
            break
    print(low, up, "low and upper prices")
    print(new_price, "new_price")
    print("=== End carbon loop")
    return new_price


def blp(
    market: xr.Dataset,
    sectors: list,
    equi: Callable[[xr.Dataset, Sequence[AbstractSector]], FindEquilibriumResults],
    commodities: list,
    new_price: float,
) -> float:
    future = market.year[-1]
    new = market.copy(deep=True)

    # Assign new carbon price
    new.prices.loc[{"year": future, "commodity": commodities}] = new_price

    new = equi(new, sectors).market

    new_emissions = (
        new.supply.sel(year=future, commodity=commodities)
        .sum(["region", "timeslice", "commodity"])
        .round(decimals=3)
    )
    print(new_price, " new_price")
    return new_emissions
