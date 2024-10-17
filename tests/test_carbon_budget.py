from unittest.mock import patch

import numpy as np
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


@patch("muse.carbon_budget.decrease_bounds")
@patch("muse.carbon_budget.increase_bounds")
@patch("muse.carbon_budget.bisect_bounds")
@patch("muse.carbon_budget.bisect_bounds_inverted")
def test_adjust_bounds(
    mock_decrease_bounds,
    mock_increase_bounds,
    mock_bisect_bounds,
    mock_bisect_bounds_inverted,
):
    from muse.carbon_budget import adjust_bounds

    lb_price = 1.0
    ub_price = 2.0
    target = 20.0

    # Test 1: lb_price_emissions < target and ub_price_emissions < target
    emissions = {lb_price: 10.0, ub_price: 15.0}
    adjust_bounds(lb_price, ub_price, emissions, target)
    mock_decrease_bounds.asset_called_once()

    # Test 2: lb_price_emissions > target and ub_price_emissions > target
    emissions = {lb_price: 25.0, ub_price: 30.0}
    adjust_bounds(lb_price, ub_price, emissions, target)
    mock_increase_bounds.asset_called_once()

    # Test 3: lb_price_emissions > target and ub_price_emissions < target
    emissions = {lb_price: 25.0, ub_price: 15.0}
    adjust_bounds(lb_price, ub_price, emissions, target)
    mock_bisect_bounds.asset_called_once()

    # Test 4: lb_price_emissions < target and ub_price_emissions > target
    emissions = {lb_price: 10.0, ub_price: 30.0}
    adjust_bounds(lb_price, ub_price, emissions, target)
    mock_bisect_bounds_inverted.asset_called_once()


def test_decrease_bounds():
    from muse.carbon_budget import decrease_bounds

    lb_price = 1.0
    ub_price = 2.0
    emissions = {lb_price: 10.0, ub_price: 30.0}
    target = 20.0
    new_lb_price, new_ub_price = decrease_bounds(lb_price, ub_price, emissions, target)
    assert new_lb_price < lb_price
    assert new_ub_price == lb_price


def test_increase_bounds():
    from muse.carbon_budget import increase_bounds

    lb_price = 1.0
    ub_price = 2.0
    emissions = {lb_price: 10.0, ub_price: 30.0}
    target = 5.0
    new_lb_price, new_ub_price = increase_bounds(lb_price, ub_price, emissions, target)
    assert new_lb_price == ub_price
    assert new_ub_price > ub_price


def test_bisect_bounds():
    from muse.carbon_budget import bisect_bounds

    lb_price = 1.0
    ub_price = 2.0
    midpoint_price = (lb_price + ub_price) / 2
    target = 20.0

    # Test 1: midpoint emissions < target
    emissions = {lb_price: 10.0, ub_price: 30.0, midpoint_price: 12.0}
    new_lb_price, new_ub_price = bisect_bounds(lb_price, ub_price, emissions, target)
    assert new_lb_price == lb_price
    assert new_ub_price == midpoint_price

    # Test 2: midpoint emissions > target
    emissions = {lb_price: 10.0, ub_price: 30.0, midpoint_price: 22.0}
    new_lb_price, new_ub_price = bisect_bounds(lb_price, ub_price, emissions, target)
    assert new_lb_price == midpoint_price
    assert new_ub_price == ub_price


def test_bisect_bounds_inverted():
    from muse.carbon_budget import bisect_bounds_inverted

    lb_price = 1.0
    ub_price = 2.0
    midpoint_price = (lb_price + ub_price) / 2
    target = 20.0

    # Test 1: midpoint emissions < target
    emissions = {lb_price: 10.0, ub_price: 30.0, midpoint_price: 12.0}
    new_lb_price, new_ub_price = bisect_bounds_inverted(
        lb_price, ub_price, emissions, target
    )
    assert new_lb_price == midpoint_price
    assert new_ub_price == ub_price

    # Test 2: midpoint emissions > target
    emissions = {lb_price: 10.0, ub_price: 30.0, midpoint_price: 22.0}
    new_lb_price, new_ub_price = bisect_bounds_inverted(
        lb_price, ub_price, emissions, target
    )
    assert new_lb_price == lb_price
    assert new_ub_price == midpoint_price
