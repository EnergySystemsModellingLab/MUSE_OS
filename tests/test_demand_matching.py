import xarray as xr
from numpy import arange
from numpy.random import choice, randint, random
from pytest import approx, fixture

from muse.demand_matching import demand_matching


def assert_matches_demand(result, demand):
    """Helper function to check if result matches demand after broadcasting."""
    actual, expected = xr.broadcast(result, demand)
    assert actual.data == approx(expected.data)


@fixture(params=["without c", "with c"])
def demand(request):
    demand = xr.DataArray(
        randint(0, 10, (5, 3, 4)),
        coords={
            "a": choice(range(15), 5, replace=False),
            "b": choice(list("abcdefghi"), 3, replace=False),
            "c": choice(["🐮", "🐄", "🕙", "📕", "💤", "🐜", "🐘"], 4, replace=False),
        },
        dims=("a", "b", "c"),
    )
    return demand.sum("c") if request.param == "without c" else demand.astype("float64")


@fixture
def cost(demand):
    return xr.DataArray(
        randint(0, 10, size=demand.shape), coords=demand.coords, dims=demand.dims
    )


@fixture
def dataset(timeslice):
    """Creates a test dataset with cost, demand, and max_production."""
    ds = xr.Dataset()

    # Create dimension coordinates
    for dim in ("d", "m", "c", "dc", "dm", "cm", "dcm"):
        nitems = randint(1, 10)
        ds[dim] = dim, choice(range(2 * nitems), nitems, replace=False)

    # Set up cost array
    shape_cost = (len(ds.c), len(ds.dc), len(ds.cm), len(ds.dcm))
    ds["cost"] = (("c", "dc", "cm", "dcm"), randint(0, 5, shape_cost).astype("float64"))

    # Set up max_production with some zero values
    shape_prod = (len(ds.m), len(ds.dm), len(ds.cm), len(ds.dcm))
    ds["max_production"] = (
        ("m", "dm", "cm", "dcm"),
        1.0 * randint(0, 10, shape_prod) * (0 == randint(0, 2, shape_prod)),
    )

    # Calculate demand based on production
    summed_production = ds.max_production.sum(("m", "cm"))

    # Create shares with correct shapes
    dc_share = xr.DataArray(random(len(ds.dc)), coords={"dc": ds.dc}, dims="dc")
    dc_share = dc_share / dc_share.sum()

    d_share = xr.DataArray(random(len(ds.d)), coords={"d": ds.d}, dims="d")
    d_share = d_share / d_share.sum()

    # Broadcast and multiply
    ds["demand"] = summed_production * dc_share * d_share

    return ds


def test_cost_order(dataset):
    """Tests the ordering of technology selection based on cost.

    This test verifies that:
    1. When costs are ordered, the lowest cost technology is selected first
    2. When two technologies have equal costs, they share the demand equally
    3. When a technology becomes more expensive, it is not selected
    """
    # Simplify dataset to focus on technology selection dimensions
    ds = dataset.sum(set(dataset.dims).difference(("dm", "cm")))
    if ds.cm.size < 2:
        return  # Skip test if insufficient technologies for comparison

    # Set up initial conditions where any tech can fulfill total demand
    total_demand = ds.demand.sum()
    ds.max_production[:] = total_demand

    # Test Case 1: Sequential cost ordering - should select first tech
    ds.cost[:] = arange(ds.cost.size).reshape(ds.cost.shape)
    result = demand_matching(ds.demand, ds.cost)
    assert_matches_demand(result.sum(ds.cost.dims), ds.demand)
    assert result.sum(ds.demand.dims)[0].data == approx(total_demand.data)

    # Test Case 2: Equal costs for first two techs - should split demand equally
    ds.cost[0] = 1
    result = demand_matching(ds.demand, ds.cost)
    assert_matches_demand(result.sum(ds.cost.dims), ds.demand)
    summed = result.sum(ds.demand.dims).data
    assert summed[0] == approx(summed[1]), (
        "First two technologies should share demand equally"
    )
    assert 2 * summed[0] == approx(total_demand.data), (
        "Sum of shared demand should equal total"
    )

    # Test Case 3: First tech more expensive - should select second tech only
    ds.cost[0] = 2
    result = demand_matching(ds.demand, ds.cost)
    assert_matches_demand(result.sum(ds.cost.dims), ds.demand)
    summed = result.sum(ds.demand.dims).data
    assert summed[1] == approx(total_demand.data), (
        "Second technology should fulfill all demand"
    )


def test_no_constraints_no_i_dims(demand, cost):
    # Test without b dimension in cost
    result = demand_matching(demand, cost.sum("b"))
    assert set(result.dims) == set(demand.dims)
    assert_matches_demand(result, demand)

    # Test with all dimensions
    result = demand_matching(demand, cost)
    assert set(result.dims) == set(demand.dims)
    assert_matches_demand(result, demand)

    # Test with summed a dimension
    result = demand_matching(demand.sum("a"), cost)
    assert set(result.dims) == set(demand.dims)
    assert_matches_demand(result.sum("a"), demand.sum("a"))


def test_no_constraints_i_dims(demand, cost):
    # Test protected dimension 'a'
    result = demand_matching(demand.sum("a"), cost, protected_dims={"a"})
    assert set(result.dims) == set(demand.dims)
    assert_matches_demand(result.sum("a"), demand.sum("a"))

    # Test protected dimension 'b'
    result = demand_matching(demand.sum("b"), cost, protected_dims={"b"})
    assert set(result.dims) == set(demand.dims)
    assert_matches_demand(result.sum("b"), demand.sum("b"))


def test_one_non_binding_constraint(demand, cost):
    """Tests constraints where excess is always 0."""
    # Test with full dimensions
    result = demand_matching(demand, cost, 2 * demand)
    assert set(result.dims) == set(demand.dims)
    assert_matches_demand(result, demand)

    # Test with summed dimension
    result = demand_matching(demand, cost, (2 * demand).sum("a"))
    assert set(result.dims) == set(demand.dims)
    assert_matches_demand(result, demand)


def test_one_cutting_constraint(demand, cost):
    """Tests constraints where excess is not always 0."""
    constraint = demand.sum("a") * 0.5

    # Test with full demand
    result = demand_matching(demand, cost, constraint)
    assert set(result.dims) == set(demand.dims)
    assert (result.sum("a") - constraint <= 1e-12).all()
    expected, actual = xr.broadcast(result.sum("a") + constraint, demand.sum("a"))
    assert actual.values == approx(expected.values)

    # Test with summed demand
    result = demand_matching(demand.sum("b"), cost, constraint)
    assert set(result.dims) == set(demand.dims)
    assert (result.sum("b") - demand.sum("b") < 1e-12).all()


def test_two_cutting_constraint(demand, cost):
    """Tests multiple constraints where excess is not always 0."""
    constraint0 = demand.sum("a") * 0.75
    constraint1 = demand.sum("b") * 0.85 + demand.sum("a") * 0.80

    # Test with full demand
    result = demand_matching(demand, cost, constraint0, constraint1)
    assert set(result.dims) == set(demand.dims)
    assert (result.sum("a") - constraint0 <= 1e-12).all()
    assert (result - constraint1 <= 1e-12).all()
    assert (result - demand <= 1e-12).all()

    # Test with summed demand
    result = demand_matching(demand.sum("a"), cost, constraint0, constraint1)
    assert set(result.dims) == set(demand.dims)
    assert (result.sum("a") - constraint0 <= 1e-12).all()
    assert (result - constraint1 <= 1e-12).all()
    assert (result.sum("a") - demand.sum("a") <= 1e-12).all()
