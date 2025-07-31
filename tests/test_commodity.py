def test_from_technologies(technologies, coords):
    """Test CommodityUsage.from_technologies method.

    Verifies that commodity usage flags are correctly set based on technology
    inputs/outputs and commodity types.
    """
    from muse.commodities import CommodityUsage

    technologies = technologies.drop_vars("comm_usage")
    technologies["commodity_type"] = "commodity", coords["commodity_type"]
    technologies = technologies.set_coords("commodity_type")

    comm_usage = CommodityUsage.from_technologies(technologies)

    # Check which commodities are used in any region/year/technology
    usage_mask = (
        technologies[["fixed_inputs", "fixed_outputs", "flexible_inputs"]] > 0
    ).any(("region", "year", "technology"))

    # Test individual commodity usage flags
    for actual, is_consumable, is_product in zip(
        comm_usage,
        usage_mask.fixed_inputs | usage_mask.flexible_inputs,
        usage_mask.fixed_outputs,
    ):
        assert bool(actual & CommodityUsage.PRODUCT) == is_product
        assert bool(actual & CommodityUsage.CONSUMABLE) == is_consumable

    # Test commodity type flags across all items
    for i in range(len(comm_usage)):
        assert (
            bool(comm_usage[i] & CommodityUsage.PRODUCT) == usage_mask.fixed_outputs[i]
        )
        assert bool(comm_usage[i] & CommodityUsage.ENVIRONMENTAL) == (
            usage_mask.commodity_type[i] == "environmental"
        )
        assert bool(comm_usage[i] & CommodityUsage.ENERGY) == (
            usage_mask.commodity_type[i] == "energy"
        )
