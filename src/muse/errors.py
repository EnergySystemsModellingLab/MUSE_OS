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
