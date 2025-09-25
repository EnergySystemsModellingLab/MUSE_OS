"""Reads and processes agent parameters from a CSV file.

This runs once per subsector, reading in a csv file and outputting a list of
dictionaries (one dictionary per agent containing the agent's parameters).
"""

from logging import getLogger
from pathlib import Path

import pandas as pd

from .helpers import read_csv


def read_agents(path: Path) -> list[dict]:
    """Reads and processes agent parameters from a CSV file."""
    df = read_agents_csv(path)
    return process_agents(df)


def read_agents_csv(path: Path) -> pd.DataFrame:
    """Reads standard MUSE agent-declaration csv-files into a DataFrame."""
    required_columns = {
        "search_rule",
        "quantity",
        "region",
        "type",
        "name",
        "agent_share",
        "decision_method",
    }
    data = read_csv(
        path,
        required_columns=required_columns,
        msg=f"Reading agent parameters from {path}.",
    )

    # Check for deprecated retrofit agents
    if "type" in data.columns:
        retrofit_agents = data[data.type.str.lower().isin(["retrofit", "retro"])]
        if not retrofit_agents.empty:
            msg = (
                "Retrofit agents will be deprecated in a future release. "
                "Please modify your model to use only agents of the 'New' type."
            )
            getLogger(__name__).warning(msg)

    # Legacy: drop AgentNumber column
    if "agent_number" in data.columns:
        data = data.drop(["agent_number"], axis=1)

    # Check consistency of objectives data columns
    objectives = [col for col in data.columns if col.startswith("objective")]
    floats = [col for col in data.columns if col.startswith("obj_data")]
    sorting = [col for col in data.columns if col.startswith("obj_sort")]

    if len(objectives) != len(floats) or len(objectives) != len(sorting):
        raise ValueError(
            "Agent objective, obj_data, and obj_sort columns are inconsistent in "
            f"{path}"
        )

    return data


def process_agents(data: pd.DataFrame) -> list[dict]:
    """Processes agent parameters DataFrame into a list of agent dictionaries."""
    result = []
    for _, row in data.iterrows():
        # Get objectives data
        objectives = (
            row[[i.startswith("objective") for i in row.index]].dropna().to_list()
        )
        sorting = row[[i.startswith("obj_sort") for i in row.index]].dropna().to_list()
        floats = row[[i.startswith("obj_data") for i in row.index]].dropna().to_list()

        # Create decision parameters
        decision_params = list(zip(objectives, sorting, floats))

        agent_type = {
            "new": "newcapa",
            "newcapa": "newcapa",
            "retrofit": "retrofit",
            "retro": "retrofit",
            "agent": "agent",
            "default": "agent",
        }[getattr(row, "type", "agent").lower()]

        # Create agent data dictionary
        data = {
            "name": row["name"],
            "region": row.region,
            "objectives": objectives,
            "search_rules": row.search_rule,
            "decision": {"name": row.decision_method, "parameters": decision_params},
            "agent_type": agent_type,
            "quantity": row.quantity,
            "share": row.agent_share,
        }

        # Add optional parameters
        if hasattr(row, "maturity_threshold"):
            data["maturity_threshold"] = row.maturity_threshold
        if hasattr(row, "spend_limit"):
            data["spend_limit"] = row.spend_limit

        # Add agent data to result
        result.append(data)

    return result
