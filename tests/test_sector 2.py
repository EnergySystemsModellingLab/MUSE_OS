from pytest import fixture, mark

<<<<<<< HEAD
pytestmark = mark.usefixtures("default_timeslice_globals")

=======
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1

@fixture
def real_market(buildings):
    from xarray import Dataset

    result = Dataset(
        {
            "timeslice": buildings.timeslices,
            "commodity": buildings.technologies.commodity,
            "region": list(set(u.region for u in buildings.agents)),
            "year": sorted(buildings.capacity.year[0:2]),
        }
    )

    def var(array, *args, low=0, high=5, dtype=float):
        from numpy.random import randint
        from xarray import DataArray

        data = randint(low, high, tuple(len(array[u]) for u in args))
        coords = {u: array[u] for u in args}
        return DataArray(data.astype(dtype), coords=coords, dims=args)

    result["prices"] = var(result, "commodity", "region", "year", "timeslice")
    result["consumption"] = var(result, "commodity", "region", "year", "timeslice")
    result["supply"] = var(result, "commodity", "region", "year", "timeslice")
    return result


@fixture(scope="function")
def mock_sector(buildings):
    """Mocked buildings."""
    from copy import deepcopy

    def mocker(agent):
        from unittest.mock import Mock, PropertyMock

        result = Mock(agent)
        type(result).assets = PropertyMock(return_value=agent.assets)
        type(result).year = PropertyMock(return_value=agent.year)
        type(result).region = PropertyMock(return_value=agent.region)
        type(result).forecast = PropertyMock(return_value=agent.forecast)
        type(result).uuid = PropertyMock(return_value=agent.uuid)
        type(result).category = PropertyMock(return_value=agent.category)
        type(result).name = PropertyMock(return_value=agent.name)
<<<<<<< HEAD
        type(result).next = Mock(return_value=None)
=======
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
        if hasattr(agent, "quantity"):
            type(result).quantity = PropertyMock(return_value=agent.quantity)
        return result

    result = deepcopy(buildings)
<<<<<<< HEAD
    result.subsectors[0].real_agents = result.subsectors[0].agents
    result.subsectors[0].agents = [mocker(u) for u in result.subsectors[0].agents]
=======
    result.real_agents = result.agents
    result.agents = [mocker(u) for u in result.agents]
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    return result


@mark.sgidata
@mark.legacy
def test_calls_to_agents(mock_sector, real_market):
    """Checks logic of calling next on sector using mocked agents."""
    from xarray import Dataset, DataArray
    from muse.outputs.sector import factory

    mock_sector.outputs = factory()
    mock_sector.next(real_market)
    for agent in mock_sector.agents:
        assert agent.next.called
        assert agent.next.call_count == 1

        args, kwargs = agent.next.call_args
        assert len(args) + len(kwargs) <= 4
        assert len(args) + len(kwargs) >= 3
        technologies = args[0] if len(args) > 0 else kwargs["technologies"]
        assert set(technologies.data_vars).issubset(mock_sector.technologies)
        market = args[1] if len(args) > 1 else kwargs["market"]
        assert isinstance(market, Dataset)
        assert "prices" in market
        assert "supply" in market
        assert "consumption" in market
        assert "capacity" in market
        demand = args[2] if len(args) > 2 else kwargs["demand"]
        assert isinstance(demand, DataArray)
        assert set(demand.squeeze().dims) == {"asset", "timeslice", "commodity"}
        assert getattr(demand, "region", agent.region) == agent.region
        if len(args) + len(kwargs) > 3:
            time_period = args[3] if len(args) > 3 else kwargs["time_period"]
            assert isinstance(time_period, int)
            assert time_period > 0


@mark.sgidata
@mark.legacy
@mark.parametrize("agent_id", range(6))
def test_call_each_agent(mock_sector, real_market, agent_id):
    """Checks logic of calling next on sector using mocked agents."""
<<<<<<< HEAD
=======

>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    from copy import deepcopy
    from muse.outputs.sector import factory

    mock_sector.outputs.sector = factory()
    mock_sector.next(real_market)

<<<<<<< HEAD
    mock = mock_sector.subsectors[0].agents[agent_id]
    real = deepcopy(mock_sector.subsectors[0].real_agents[agent_id])
    assert mock.next.call_count == 1
=======
    mock = mock_sector.agents[agent_id]
    real = deepcopy(mock_sector.real_agents[agent_id])
    assert mock.next.call_count == 1
    real.investments = "normalized"
>>>>>>> 44e9eaf3c2493e9a0ac61be1c74061027052e6c1
    real.next(*mock.next.call_args[0], **mock.next.call_args[1])
