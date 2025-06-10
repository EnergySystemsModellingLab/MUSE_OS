import numpy as np
from numpy.random import choice, rand
from pytest import approx, fixture
from scipy.stats import rankdata
from xarray import Dataset

from muse.decisions import (
    epsilon_constraints,
    lexical_comparison,
    retro_epsilon_constraints,
    single_objective,
    weighted_sum,
)


@fixture
def objectives():
    objectives = Dataset()
    objectives["replacement"] = choice(
        list("abcdefghijklmnopqrstuvwxyz"), 10, replace=False
    )
    objectives["asset"] = choice(objectives.replacement.values, 5, replace=False)
    objectives["other"] = choice(
        ["what", "where", "when", "why", "how"], 3, replace=False
    )

    objectives["a"] = add_var(objectives, "replacement")
    objectives["b"] = add_var(objectives, "asset", "replacement")
    objectives["c"] = add_var(objectives, "replacement", "other")

    return objectives


def add_var(coordinates, *dims, factor=100.0):
    shape = tuple(len(coordinates[u]) for u in dims)
    return dims, (rand(*shape) * factor).astype(type(factor))


def test_weighted_sum(objectives):
    weights = {"a": -1, "c": 2}

    def normalize(objective):
        return objective / abs(objective).max("replacement")

    expected = (
        normalize(objectives.a) * weights["a"]
        + normalize(objectives.b)
        + normalize(objectives.c).mean("other") * weights["c"]
    )

    actual = weighted_sum(objectives, weights)
    assert actual.values == approx(expected.values)


def test_lexical():
    """Test lexical comparison against hand-constructed tuples."""
    shape = (5, 10)
    a = rand(*shape) * 10 - 5
    b = rand(*shape) * 10 - 5
    c = rand(*shape) * 10 - 5

    parameters = [("b", -rand() * 0.1), ("a", rand() * 0.1), ("c", rand())]
    param_dict = dict(parameters)

    mina = (a * param_dict["a"]).min(1)
    minb = -(b * abs(param_dict["b"])).max(1)
    minc = (c * param_dict["c"]).min(1)

    expected = np.zeros(shape=shape, dtype=object)
    for i in range(shape[0]):
        for j in range(shape[1]):
            expected[i, j] = (
                int(np.floor(b[i, j] / minb[i])),
                int(np.floor(a[i, j] / mina[i])),
                c[i, j] / minc[i],
            )

    objectives = Dataset(
        {
            "a": (("asset", "replacement"), a),
            "b": (("asset", "replacement"), b),
            "c": (("asset", "replacement"), c),
        }
    )
    objectives["asset"] = choice(objectives.replacement, shape[0], replace=False)

    actual = lexical_comparison(objectives, parameters)
    assert actual.shape == expected.shape
    for i in range(shape[0]):
        assert actual.values[i] == approx(rankdata(expected[i]))


def test_epsilon_constraints(objectives):
    def reshape_array(size, shape, start=1):
        return np.array(range(start, size + start)).reshape(shape)

    objectives.b[:] = reshape_array(objectives.b.size, objectives.b.shape)
    objectives.c[:] = reshape_array(objectives.c.size, objectives.c.shape)

    # Test case 1: Basic constraints
    expected = objectives.a * (objectives.asset == objectives.asset)
    params = [("a", True), ("b", True, objectives.b.max() + 1)]
    actual = epsilon_constraints(objectives, params)
    assert actual.values == approx(expected.values)

    # Test case 2: Negative constraints
    expected = -objectives.a * (objectives.asset == objectives.asset)
    params = [("a", False), ("b", True, objectives.b.max() + 1)]
    actual = epsilon_constraints(objectives, params)
    assert actual.values == approx(expected.values)

    # Test case 3: Binary choice constraints
    objectives.b[:] = choice((1, 2), objectives.b.size).reshape(objectives.b.shape)
    params = [("a", True), ("b", True, 1.5)]
    expected = objectives.a.where(objectives.b == 1).fillna(-1)
    actual = epsilon_constraints(objectives, params, mask=-1)
    assert actual.values == approx(expected.values)

    # Test case 4: Multiple constraints
    objectives.b[:] = choice((1, 2), objectives.b.size).reshape(objectives.b.shape)
    objectives.c[:] = choice((1, 2, 3), objectives.c.size).reshape(objectives.c.shape)
    params = [("a", True), ("b", True, 1.5), ("c", False, 1.2)]
    condition = (objectives.b == 1) & (objectives.c >= 2).all("other")
    expected = objectives.a.where(condition).fillna(-1)
    actual = epsilon_constraints(objectives, params, mask=-1)
    assert (np.isnan(actual) == np.isnan(expected)).all()
    assert actual.fillna(0).values == approx(expected.fillna(0).values)

    # Test retro_epsilon_constraints
    expected = expected.fillna(-1)
    while not np.isnan(expected).any():
        objectives.b[:] = choice((1, 2), objectives.b.size).reshape(objectives.b.shape)
        objectives.c[:] = choice((1, 2, 3), objectives.c.size).reshape(
            objectives.c.shape
        )
        condition = (objectives.b == 1) & (objectives.c >= 2).all("other")
        expected = objectives.a.where(condition).min("replacement")
    actual = retro_epsilon_constraints(objectives, params)
    assert not np.isnan(actual).any()


def test_single_objectives(objectives):
    assert single_objective(objectives, "a").values == approx(objectives.a.values)
    assert single_objective(objectives, ["b", False]).values == approx(
        -objectives.b.values
    )


if __name__ == "__main__":
    test_objectives = objectives()  # For debugging purposes
