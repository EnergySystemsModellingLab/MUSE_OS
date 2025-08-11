from __future__ import annotations


def test_read_technodictionary_csv(default_model_path):
    """Test reading the technodictionary CSV file."""
    from muse.readers.csv.technologies import read_technodictionary_csv

    technodictionary_path = default_model_path / "power" / "Technodata.csv"
    technodictionary_df = read_technodictionary_csv(technodictionary_path)
    assert technodictionary_df is not None
    mandatory_columns = {
        "cap_exp",
        "region",
        "var_par",
        "fix_exp",
        "interest_rate",
        "utilization_factor",
        "minimum_service_factor",
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
    }
    assert set(technodictionary_df.columns) == mandatory_columns | extra_columns


def test_read_technodata_timeslices_csv(default_timeslice_model_path):
    """Test reading the technodata timeslices CSV file."""
    from muse.readers.csv.technologies import read_technodata_timeslices_csv

    timeslices_path = (
        default_timeslice_model_path / "power" / "TechnodataTimeslices.csv"
    )
    timeslices_df = read_technodata_timeslices_csv(timeslices_path)
    mandatory_columns = {
        "utilization_factor",
        "technology",
        "minimum_service_factor",
        "region",
        "year",
    }
    extra_columns = {
        "month",
        "hour",
        "day",
    }
    assert set(timeslices_df.columns) == mandatory_columns | extra_columns


def test_read_existing_capacity_csv(default_model_path):
    """Test reading the initial capacity CSV file."""
    from muse.readers.csv.assets import read_existing_capacity_csv

    capacity_path = default_model_path / "power" / "ExistingCapacity.csv"
    capacity_df = read_existing_capacity_csv(capacity_path)
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
        "2040",
        "2020",
    }
    assert set(capacity_df.columns) == mandatory_columns | extra_columns


def test_read_global_commodities_csv(default_model_path):
    """Test reading the global commodities CSV file."""
    from muse.readers.csv.commodities import read_global_commodities_csv

    commodities_path = default_model_path / "GlobalCommodities.csv"
    commodities_df = read_global_commodities_csv(commodities_path)
    mandatory_columns = {
        "commodity",
        "commodity_type",
    }
    extra_columns = {"unit"}
    assert set(commodities_df.columns) == mandatory_columns | extra_columns


def test_read_timeslice_shares_csv(default_correlation_model_path):
    """Test reading the timeslice shares CSV file."""
    from muse.readers.csv.correlation_consumption import read_timeslice_shares_csv

    shares_path = (
        default_correlation_model_path
        / "residential_presets"
        / "TimesliceSharepreset.csv"
    )
    shares_df = read_timeslice_shares_csv(shares_path)
    mandatory_columns = {
        "region",
        "timeslice",
    }
    extra_columns = {
        "electricity",
        "heat",
        "wind",
        "gas",
        "CO2f",
    }
    assert set(shares_df.columns) == mandatory_columns | extra_columns


def test_read_agents_csv(default_model_path):
    """Test reading the agent parameters CSV file."""
    from muse.readers.csv.agents import read_agents_csv

    agents_path = default_model_path / "Agents.csv"
    agents_df = read_agents_csv(agents_path)
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
        "obj_sort1",
        "objective1",
        "obj_data1",
    }
    assert set(agents_df.columns) == mandatory_columns | extra_columns


def test_read_macro_drivers_csv(default_correlation_model_path):
    """Test reading the macro drivers CSV file."""
    from muse.readers.csv.correlation_consumption import read_macro_drivers_csv

    macro_path = (
        default_correlation_model_path / "residential_presets" / "Macrodrivers.csv"
    )
    macro_df = read_macro_drivers_csv(macro_path)
    mandatory_columns = {
        "region",
        "variable",
    }
    extra_columns = {
        *[str(year) for year in range(2010, 2111)],
    }
    assert set(macro_df.columns) == mandatory_columns | extra_columns

    assert "Population" in macro_df["variable"].values
    assert "GDP|PPP" in macro_df["variable"].values


def test_read_projections_csv(default_model_path):
    """Test reading the projections CSV file."""
    from muse.readers.csv.market import read_projections_csv

    projections_path = default_model_path / "Projections.csv"
    projections_df = read_projections_csv(projections_path)
    mandatory_columns = {
        "year",
        "attribute",
        "region",
    }
    extra_columns = {
        "CO2f",
        "electricity",
        "gas",
    }
    assert set(projections_df.columns) == mandatory_columns | extra_columns


def test_read_regression_parameters_csv(default_correlation_model_path):
    """Test reading the regression parameters CSV file."""
    from muse.readers.csv.correlation_consumption import read_regression_parameters_csv

    regression_path = (
        default_correlation_model_path
        / "residential_presets"
        / "regressionparameters.csv"
    )
    regression_df = read_regression_parameters_csv(regression_path)
    mandatory_columns = {
        "sector",
        "region",
        "function_type",
        "coeff",
    }
    extra_columns = {
        "CO2f",
        "electricity",
        "gas",
        "heat",
    }
    assert set(regression_df.columns) == mandatory_columns | extra_columns


def test_read_presets_csv(default_model_path):
    """Test reading the presets CSV files."""
    from muse.readers.csv.presets import read_presets_csv

    presets_path = (
        default_model_path / "residential_presets" / "Residential2020Consumption.csv"
    )
    presets_df = read_presets_csv(presets_path)

    mandatory_columns = {
        "region",
        "timeslice",
    }
    extra_columns = {
        "heat",
    }
    assert set(presets_df.columns) == mandatory_columns | extra_columns


def test_read_existing_trade_csv(trade_model_path):
    """Test reading the existing trade CSV file."""
    from muse.readers.csv.trade_assets import read_existing_trade_csv

    trade_path = trade_model_path / "power" / "ExistingTrade.csv"
    trade_df = read_existing_trade_csv(trade_path)
    mandatory_columns = {
        "region",
        "technology",
        "year",
    }
    extra_columns = {"r1", "r2"}
    assert set(trade_df.columns) == mandatory_columns | extra_columns


def test_read_trade_technodata(trade_model_path):
    """Test reading the trade technodata CSV file."""
    from muse.readers.csv.trade_technodata import read_trade_technodata_csv

    trade_path = trade_model_path / "power" / "TradeTechnodata.csv"
    trade_df = read_trade_technodata_csv(trade_path)
    mandatory_columns = {"technology", "region", "parameter"}
    extra_columns = {"r1", "r2"}
    assert set(trade_df.columns) == mandatory_columns | extra_columns
