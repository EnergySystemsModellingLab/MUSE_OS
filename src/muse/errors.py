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
    """Indicates that there is a conflict in the commodity units between files."""

    msg = """The units of “CommIn” “CommOut” and “GlobalCommodities” files must be the
same, including the casing. Check the consistency of the units across those three files.
"""

    def __str__(self):
        return self.msg


class GrowthOfCapacityTooConstrained(Exception):
    """Indicates that the investment step failed because capacity could not grow."""

    msg = """Error during the investment process. The capacity was not allowed to grow
sufficiently in order to match the demand. Consider increasing the MaxCapacityAddition
and/or the MaxCapacityGrowth in the technodata."""

    def __str__(self):
        return self.msg


class TechnologyNotDefined(Exception):
    """Indicates that the initialisation fails because a technology is not found."""

    msg = """Error during the initialisation of a sector.
The model tries to assign a share of the total capacity to an agent but it
cannot find a match between technodata and existing capacity.
Check the spelling of the technology names."""

    def __str__(self):
        return self.msg


class FailedInterpolation(Exception):
    """Indicates that the initialisation fails due to interpolation."""

    msg = """Error during the initialisation of a sector.
The model tries to interpolate values in time of the technologies.
It fails to interpolate some parameters for selected years.
This results in nans in the datasets of technology data.
Check the spelling of the technology names in the sector data."""

    def __str__(self):
        return self.msg


class RetrofitAgentInStandardDemandShare(Exception):
    msg = """A retrofit agent has been found in a 'New agents'-only demand share
function. Make sure you remove all the retro agents from the Agents input files or use a
demand share method that can handle both new and retro agents."""

    def __str__(self):
        return self.msg


class AgentWithNoAssetsInDemandShare(Exception):
    msg = """This error refers to an agent with no assets. To fix this error, check the
capacity assignment to the agents. One possibility is that you have decided not
to use "Retrofit" agents, as such you may have already removed them from the
agent definition file and the file of technodata, the system TOML file should
change the demand_share to "standard_demand" function in each subsector
section for each of the selected sectors to model."""

    def __str__(self):
        return self.msg


class NoInteractionsFound(Exception):
    msg = """A network with no interactions has been found. This might be the case if
there are no retrofit agents and yet a 'new_to_retro' network has been defined for a
particular sector. Asses the existence of both new and retrofit agents for all sectors
and remove the new_to_retro interacton network if it is not needed """

    def __str__(self):
        return self.msg
