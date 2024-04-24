from pytest import approx, fixture


@fixture
def objectives():
    from numpy.random import choice
    from xarray import Dataset

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
    from numpy.random import rand

    shape = tuple(len(coordinates[u]) for u in dims)
    return dims, (rand(*shape) * factor).astype(type(factor))


def test_weighted_sum(objectives):
    from muse.decisions import weighted_sum

    weights = {"a": -1, "c": 2}

    def normalize(objective):
        min, max = objective.min("replacement"), objective.max("replacement")
        return (objective - min) / (max - min)

    expected = (
        normalize(objectives.a) * weights["a"]
        + normalize(objectives.b)
        + normalize(objectives.c).mean("other") * weights["c"]
    )

    actual = weighted_sum(objectives, weights)
    assert actual.values == approx(expected.values)


def test_lexical():
    """Test lexical comparison against hand-constructed tuples."""
    from muse.decisions import lexical_comparison
    from numpy import floor, zeros
    from numpy.random import choice, rand
    from scipy.stats import rankdata
    from xarray import Dataset

    a = rand(5, 10) * 10 - 5
    b = rand(5, 10) * 10 - 5
    c = rand(5, 10) * 10 - 5

    parameters = [("b", -rand() * 0.1), ("a", rand() * 0.1), ("c", rand())]
    mina = (a * dict(parameters)["a"]).min(1)
    minb = -(b * abs(dict(parameters)["b"])).max(1)
    minc = (c * dict(parameters)["c"]).min(1)

    expected = zeros(shape=a.shape, dtype=object)
    for i in range(expected.shape[0]):
        for j in range(expected.shape[1]):
            expected[i, j] = (
                int(floor(b[i, j] / minb[i])),
                int(floor(a[i, j] / mina[i])),
                c[i, j] / minc[i],
            )

    objectives = Dataset()
    objectives["a"] = ("asset", "replacement"), a
    objectives["b"] = ("asset", "replacement"), b
    objectives["c"] = ("asset", "replacement"), c
    objectives["asset"] = choice(
        objectives.replacement, len(objectives.asset), replace=False
    )

    actual = lexical_comparison(objectives, parameters)
    assert actual.shape == expected.shape
    for i in range(expected.shape[0]):
        assert actual.values[i] == approx(rankdata(expected[i]))


def test_epsilon_constraints(objectives):
    from muse.decisions import epsilon_constraints, retro_epsilon_constraints
    from numpy import array, isnan
    from numpy.random import choice

    objectives.b[:] = array(range(1, objectives.b.size + 1)).reshape(objectives.b.shape)
    objectives.c[:] = array(range(1, objectives.c.size + 1)).reshape(objectives.c.shape)

    expected = objectives.a * (objectives.asset == objectives.asset)
    parameters = [("a", True), ("b", True, objectives.b.max() + 1)]
    actual = epsilon_constraints(objectives, parameters)
    assert actual.values == approx(expected.values)

    expected = -objectives.a * (objectives.asset == objectives.asset)
    parameters = [("a", False), ("b", True, objectives.b.max() + 1)]
    actual = epsilon_constraints(objectives, parameters)
    assert actual.values == approx(expected.values)

    objectives.b[:] = choice((1, 2), objectives.b.size).reshape(objectives.b.shape)
    parameters = [("a", True), ("b", True, 1.5)]
    expected = objectives.a.where(objectives.b == 1).fillna(-1)
    actual = epsilon_constraints(objectives, parameters, mask=-1)
    assert actual.values == approx(expected.values)

    objectives.b[:] = choice((1, 2), objectives.b.size).reshape(objectives.b.shape)
    objectives.c[:] = choice((1, 2, 3), objectives.c.size).reshape(objectives.c.shape)
    parameters = [("a", True), ("b", True, 1.5), ("c", False, 1.2)]
    condition = (objectives.b == 1) & (objectives.c >= 2).all("other")
    expected = objectives.a.where(condition).fillna(-1)
    actual = epsilon_constraints(objectives, parameters, mask=-1)
    assert (isnan(actual) == isnan(expected)).all()
    assert actual.fillna(0).values == approx(expected.fillna(0).values)

    # retro_ should tweak the parameters so that assets are always
    # included. Hence expected values cannot be nan.
    expected = expected.fillna(-1)
    while not isnan(expected).any():
        objectives.b[:] = choice((1, 2), objectives.b.size).reshape(objectives.b.shape)
        objectives.c[:] = choice((1, 2, 3), objectives.c.size).reshape(
            objectives.c.shape
        )
        condition = (objectives.b == 1) & (objectives.c >= 2).all("other")
        expected = objectives.a.where(condition).min("replacement")
    actual = retro_epsilon_constraints(objectives, parameters)
    assert not isnan(actual).any()


def test_single_objectives(objectives):
    from muse.decisions import single_objective

    assert single_objective(objectives, "a").values == approx(objectives.a.values)
    actual = single_objective(objectives, ["b", False])
    assert actual.values == approx(-objectives.b.values)


# when developping/debugging, these few lines help setup the input for the
# different tests
if __name__ == "main":
    # fmt: off
    from tests.agents import test_objectives  # noqa
    objectives = test_decisions.objectives()  # noqa
