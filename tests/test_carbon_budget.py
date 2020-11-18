import numpy as np
import xarray as xr
from pytest import approx, fixture


@fixture
def linear_sampling():
    from muse.carbon_budget import linear_fun

    a = 0.01
    budget = 420000
    carbon_price = 0.02
    b = (carbon_price - a) / budget

    emissions = np.random.uniform(low=budget * 0.75, high=budget * 1.25, size=4)
    prices = linear_fun(emissions, a, b)

    return prices, emissions, budget, a, b


@fixture
def exp_sampling():
    from muse.carbon_budget import exponential_fun

    c = 0.01
    budget = 420000
    b = 1 / budget
    carbon_price = 0.02
    a = (carbon_price - c) / np.exp(-1)

    emissions = np.random.uniform(low=budget * 0.75, high=budget * 1.25, size=4)
    prices = exponential_fun(emissions, a, b, c)

    return prices, emissions, budget, a, b, c


def test_create_sample():
    from muse.carbon_budget import create_sample

    carbon_price = 0.02
    budget = 420000
    size = 4

    current_emissions = 300000
    expected = np.array([0.02, 0.01933333, 0.01866667, 0.018])
    sample = create_sample(carbon_price, current_emissions, budget, size)
    assert sample == approx(expected)

    current_emissions = 500000
    expected = np.array([0.02, 0.02115623, 0.02231246, 0.02346869])
    sample = create_sample(carbon_price, current_emissions, budget, size)
    assert sample == approx(expected)


def test_linear_guess_and_weights(linear_sampling):
    from muse.carbon_budget import linear_guess_and_weights

    (prices, emissions, budget, a, b) = linear_sampling

    actual_weights = np.abs(emissions - budget)

    (est_a, est_b), est_weights = linear_guess_and_weights(prices, emissions, budget)

    assert actual_weights == approx(est_weights)
    assert a == approx(est_a)
    assert b == approx(est_b)


def test_linear(linear_sampling):
    from muse.carbon_budget import linear, linear_fun

    (prices, emissions, budget, a, b) = linear_sampling

    actual_price = linear_fun(budget, a, b)
    est_price = linear(prices, emissions, budget)
    assert actual_price == approx(est_price)


def test_exp_guess_and_weights(exp_sampling):
    from muse.carbon_budget import exp_guess_and_weights

    (prices, emissions, budget, a, b, c) = exp_sampling

    actual_weights = np.abs(emissions - budget)

    (est_a, est_b, est_c), est_weights = exp_guess_and_weights(
        prices, emissions, budget
    )

    assert actual_weights == approx(est_weights)
    assert a == approx(est_a, rel=1e3)
    assert b == approx(est_b, rel=1e3)
    assert c == approx(est_c, rel=1e3)


def test_exponential(exp_sampling):
    from muse.carbon_budget import exponential, exponential_fun

    (prices, emissions, budget, a, b, c) = exp_sampling

    actual_price = exponential_fun(budget, a, b, c)

    est_price = exponential(prices, emissions, budget)
    assert actual_price == approx(est_price)


def test_overshoot():
    from muse.carbon_budget import update_carbon_budget

    year = 0
    carbonbudget = np.array([4e6, 5e6, 6e6], dtype=int)

    emissions = int(5.3e6)
    expected = int(5.7e6)
    actual = update_carbon_budget(carbonbudget, emissions, year, over=True, under=False)
    assert expected == actual

    emissions = int(4e6)
    expected = int(6e6)
    actual = update_carbon_budget(carbonbudget, emissions, year, over=True, under=False)
    assert expected == actual


def test_undershoot():
    from muse.carbon_budget import update_carbon_budget

    year = 0
    carbonbudget = np.array([4e6, 5e6, 6e6], dtype=int)

    emissions = int(4.7e6)
    expected = int(6.3e6)
    actual = update_carbon_budget(carbonbudget, emissions, year, over=False, under=True)
    assert expected == actual

    emissions = int(6e6)
    expected = int(6e6)
    actual = update_carbon_budget(carbonbudget, emissions, year, over=False, under=True)
    assert expected == actual


def test_refine_new_price(market):
    from muse.carbon_budget import refine_new_price

    num_years = 5
    years = np.linspace(2010, 2020, num_years, dtype=int)
    commodities = ["CH4", "CO2"]
    budget = xr.DataArray(
        np.linspace(4e6, 6e6, num_years), dims=["year"], coords={"year": years}
    )
    price_too_high_threshold = 10

    market = market.interp(year=years)
    market["prices"] = market.prices.mean("timeslice")
    future = years[2]
    price = market.prices.sel(year=future, commodity=commodities).mean(
        ["region", "commodity"]
    )
    sample = np.linspace(price, 4 * price, 4)

    carbon_price = market.prices.sel(
        year=market.year < future, commodity=commodities
    ).mean(["region", "commodity"])
    too_high = price_too_high_threshold * max(min(carbon_price.values), 0.1)

    # Checking price too high
    price = 1.1 * too_high
    actual = refine_new_price(
        market,
        carbon_price,
        budget,
        sample,
        price,
        commodities,
        price_too_high_threshold,
    )
    assert actual < price

    # Just fine
    price = 0.9 * too_high
    actual = refine_new_price(
        market,
        carbon_price,
        budget,
        sample,
        price,
        commodities,
        price_too_high_threshold,
    )
    assert actual == price

    # Negative
    price = -price
    actual = refine_new_price(
        market,
        carbon_price,
        budget,
        sample,
        price,
        commodities,
        price_too_high_threshold,
    )
    assert actual > 0
