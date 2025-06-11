import pytest

from muse import examples
from muse.readers.csv import (
    read_agent_parameters_csv,
    read_global_commodities_csv,
    read_initial_assets_csv,
    read_initial_market_csv,
    read_presets_csv,
    read_technodata_timeslices_csv,
    read_technodictionary_csv,
    read_technologies_csv,
)


@pytest.fixture
def model_path(tmp_path):
    """Creates temporary folder containing the default model."""
    examples.copy_model(name="default", path=tmp_path)
    return tmp_path / "model"


@pytest.fixture
def timeslice_model_path(tmp_path):
    """Creates temporary folder containing the default model."""
    examples.copy_model(name="default_timeslice", path=tmp_path)
    return tmp_path / "model"


def test_read_technodictionary_csv(model_path):
    """Test reading the technodictionary CSV file."""
    technodictionary_path = model_path / "power" / "Technodata.csv"
    technodictionary_df = read_technodictionary_csv(technodictionary_path)
    assert technodictionary_df is not None
    mandatory_columns = {
        "cap_exp",
        "region",
        "var_par",
        "fix_exp",
        "interest_rate",
        "utilization_factor",
        "scaling_size",
        "year",
        "cap_par",
        "var_exp",
        "technology",
        "technical_life",
        "fix_par",
    }
    extra_columns = {
        "agent1",
        "max_capacity_growth",
        "total_capacity_limit",
        "max_capacity_addition",
        "type",
        "efficiency",
    }
    assert set(technodictionary_df.columns) == mandatory_columns | extra_columns


def test_read_technodata_timeslices_csv(timeslice_model_path):
    """Test reading the technodata timeslices CSV file."""
    timeslices_path = timeslice_model_path / "power" / "TechnodataTimeslices.csv"
    timeslices_df = read_technodata_timeslices_csv(timeslices_path)
    assert timeslices_df is not None
    mandatory_columns = {
        "utilization_factor",
        "technology",
        "minimum_service_factor",
        "region",
    }
    extra_columns = {
        "month",
        "hour",
        "year",
        "day",
    }
    assert set(timeslices_df.columns) == mandatory_columns | extra_columns


def test_read_technologies_csv(timeslice_model_path):
    """Test reading the technologies CSV files."""
    technodata_path = timeslice_model_path / "power" / "Technodata.csv"
    comm_out_path = timeslice_model_path / "power" / "CommOut.csv"
    comm_in_path = timeslice_model_path / "power" / "CommIn.csv"
    timeslices_path = timeslice_model_path / "power" / "TechnodataTimeslices.csv"

    technodata_df, comm_out_df, comm_in_df, timeslices_df = read_technologies_csv(
        technodata_path, comm_out_path, comm_in_path, timeslices_path
    )

    assert technodata_df is not None
    assert comm_out_df is not None
    assert comm_in_df is not None
    assert timeslices_df is not None

    # Check required columns
    for df in [technodata_df, comm_out_df, comm_in_df]:
        assert "technology" in df.columns
        assert "region" in df.columns
        assert "year" in df.columns


def test_read_initial_assets_csv(model_path):
    """Test reading the initial assets CSV file."""
    assets_path = model_path / "power" / "ExistingCapacity.csv"
    assets_df = read_initial_assets_csv(assets_path)
    assert assets_df is not None
    mandatory_columns = {
        "region",
        "technology",
    }
    extra_columns = {
        "2025",
        "2030",
        "2035",
        "2050",
        "2045",
        "unit",
        "2040",
        "2020",
    }
    assert set(assets_df.columns) == mandatory_columns | extra_columns


def test_read_global_commodities_csv(model_path):
    """Test reading the global commodities CSV file."""
    commodities_path = model_path / "GlobalCommodities.csv"
    commodities_df = read_global_commodities_csv(commodities_path)
    assert commodities_df is not None
    mandatory_columns = {
        "commodity",
        "comm_type",
    }
    extra_columns = {"heat_rate", "unit", "commodity_emission_factor_CO2"}
    assert set(commodities_df.columns) == mandatory_columns | extra_columns


# def test_read_timeslice_shares_csv(model_path):
#     """Test reading the timeslice shares CSV file."""
#     shares_path = model_path / "power" / "TimesliceShares.csv"
#     shares_df = read_timeslice_shares_csv(shares_path)
#     assert shares_df is not None
#     assert "region" in shares_df.columns
#     assert "timeslice" in shares_df.columns


def test_read_agent_parameters_csv(model_path):
    """Test reading the agent parameters CSV file."""
    agents_path = model_path / "Agents.csv"
    agents_df = read_agent_parameters_csv(agents_path)
    assert agents_df is not None
    mandatory_columns = {
        "search_rule",
        "quantity",
        "region",
        "type",
        "name",
        "agent_share",
        "decision_method",
    }
    extra_columns = {
        "objsort2",
        "objective3",
        "maturity_threshold",
        "objsort1",
        "spend_limit",
        "obj_data3",
        "objective2",
        "objective1",
        "obj_data2",
        "objsort3",
        "obj_data1",
    }
    assert set(agents_df.columns) == mandatory_columns | extra_columns


# def test_read_macro_drivers_csv(model_path):
#     """Test reading the macro drivers CSV file."""
#     macro_path = model_path / "power" / "MacroDrivers.csv"
#     macro_df = read_macro_drivers_csv(macro_path)
#     assert macro_df is not None
#     assert "region" in macro_df.columns
#     assert "variable" in macro_df.columns
#     assert "Population" in macro_df["variable"].values
#     assert "GDP|PPP" in macro_df["variable"].values


def test_read_initial_market_csv(model_path):
    """Test reading the initial market CSV files."""
    projections_path = model_path / "Projections.csv"
    projections_df, import_df, export_df = read_initial_market_csv(projections_path)
    assert projections_df is not None
    mandatory_columns = {
        "year",
        "attribute",
        "region",
    }
    extra_columns = {
        "CO2f",
        "electricity",
        "wind",
        "gas",
        "heat",
    }
    assert set(projections_df.columns) == mandatory_columns | extra_columns


# def test_read_regression_parameters_csv(model_path):
#     """Test reading the regression parameters CSV file."""
#     regression_path = model_path / "power" / "RegressionParameters.csv"
#     regression_df = read_regression_parameters_csv(regression_path)
#     assert regression_df is not None
#     assert "sector" in regression_df.columns
#     assert "region" in regression_df.columns
#     assert "function_type" in regression_df.columns
#     assert "coeff" in regression_df.columns


def test_read_presets_csv(model_path):
    """Test reading the presets CSV files."""
    presets_path = model_path / "residential_presets" / "*.csv"
    presets_dict = read_presets_csv(presets_path)
    assert presets_dict is not None
    assert len(presets_dict) > 0

    # Check first year's data
    first_year = min(presets_dict.keys())
    first_year_df = presets_dict[first_year]
    mandatory_columns = {
        "region",
        "timeslice",
    }
    extra_columns = {
        "heat",
        "electricity",
        "wind",
        "gas",
        "CO2f",
    }
    assert set(first_year_df.columns) == mandatory_columns | extra_columns
