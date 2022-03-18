class RetrofitAgentNotDefined(Exception):
    """Indicates that the retrofit agent has not been defined."""

    msg = """The retrofit agent has not been defined. This might be because it actually
has not been defined in the agents file or because there is a typo in its name in either
the agents and or technodata files."""

    def __str__(self):
        return self.msg
