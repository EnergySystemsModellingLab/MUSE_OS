class RetrofitAgentNotDefined(Exception):
    """Indicates that the retrofit agent has not been defined."""

    msg = """The retrofit agent has not been defined. This might be because it actually
has not been defined in the agents file or specified in the technodata file or it cannot
be found because there is a typo in its name in either the agents and or technodata
files. When a retrofit agent is defined in the agent file, the same name should be
reported as an additional column in the technodata, assigning the fraction of the
initial stock assigned to each retrofit agent. In presence of multiple retrofit agents,
a new column needs to be added per agent."""

    def __str__(self):
        return self.msg


class UnitsConflictInCommodities(Exception):
    """Indicates that there is a conflcit in the commodity units between files."""

    msg = """The units of “CommIn” “CommOut” and “GlobalCommodities” files must be the
same, including the casing. Check the consistency of the units across those thre files.
"""

    def __str__(self):
        return self.msg


class GrowthOfCapacityTooConstrained(Exception):
    """Indicates that the investment step failed because capacity could not grow."""

    msg = """Error during the investment process. The capacity was not allowed to grow
sufficiently in order to match the demand. Consider increating the MaxCapacityAddition
and/or the MaxCapacityGrowth in the technodata."""

    def __str__(self):
        return self.msg


class TechnologyNotDefined(Exception):
    """Indicates that the initialisation fails because a technology is not found"""

    msg = """Error during the initialisation of a sector.
The model tries to assign a share of the total capacity to an agent but it
cannot find a match between technodata and existing capacity.
Check the spelling of your technology name."""

    def __str__(self):
        return self.msg
