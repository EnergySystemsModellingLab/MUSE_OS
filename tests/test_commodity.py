def test_from_technologies(technologies, coords):
    from muse.commodities import CommodityUsage

    technologies = technologies.drop_vars("comm_usage")
    technologies["comm_type"] = "commodity", coords["comm_type"]
    technologies = technologies.set_coords("comm_type")
    comm_usage = CommodityUsage.from_technologies(technologies)

    redux = (
        technologies[["fixed_inputs", "fixed_outputs", "flexible_inputs"]] > 0
    ).any(("region", "year", "technology"))

    for actual, is_cons, is_prod in zip(
        comm_usage, redux.fixed_inputs | redux.flexible_inputs, redux.fixed_outputs
    ):
        assert bool(actual & CommodityUsage.PRODUCT) == is_prod
        assert bool(actual & CommodityUsage.CONSUMABLE) == is_cons

    assert ((comm_usage & CommodityUsage.PRODUCT != 0) == redux.fixed_outputs).all()
    assert (
        (comm_usage & CommodityUsage.ENVIRONMENTAL != 0)
        == (redux.comm_type == "environmental")
    ).all()

    assert (
        (comm_usage & CommodityUsage.ENERGY != 0) == (redux.comm_type == "energy")
    ).all()
