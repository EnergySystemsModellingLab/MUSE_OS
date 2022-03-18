class RetrofitAgentNotDefined(Exception):
    """Indicates that the retrofit agent has not been defined."""

    msg = """The retrofit agent has not been defined. This might be because it actually
has not been defined in the agents file or because it cannot be found because there is a
typo in its name in either the agents and or technodata files."""

    def __str__(self):
        return self.msg


class UnitsConflictInCommodities(Exception):
    """Indictaes that there is a conflcit in the commodity units between files."""

    msg = """The units of “CommIn” “CommOut” and “GlobalCommodities” files must be the
same, including the casing. Check the consistency of the units across those thre files.
"""

    def __str__(self):
        return self.msg


class GrowthOfCapacityTooConstrained(Exception):
    """Indicataes that the investment step failed because capacity could not grow."""

    msg = """Error during the investment process. The capacity was not allowed to grow
suficiently in order to match the demand. Consider increating the MaxCapacityAddition
and/or the MaxCapacityGrowth in the technodata."""

    def __str__(self):
        return self.msg
