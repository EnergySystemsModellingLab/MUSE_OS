# Starting point: copy default model

"""
ADD A NEW COMMODITY

commodity_name: solar
sector: power

copy_from: wind

"""

# File: technodata/power/CommIn.csv
# Add new column {commodity_name} as copy of {copy_from}
# >>> No values changed

# File: technodata/{sector}/CommOut.csv
# Add new column {commodity_name} as copy of {copy_from}
# >>> No values changed

# File: input/BaseYearImport.csv
# Add new column {commodity_name} as copy of {copy_from}
# >>> No values changed

# File: input/BaseYearExport.csv
# Add new column {commodity_name} as copy of {copy_from}
# >>> No values changed

# File: input/GlobalCommodities.csv
# Copy row(s) where Commodity={copy_from.capitalize()}, changing Commidity to
#     {commodity_name.capitalize()} and CommodityName to {commodity_name}
# >>> No values changed

# File: input/Projections.csv
# Add new column {commodity_name} as copy of {copy_from}
# >>> No values changed

"""
ADD A NEW PROCESS

process_name: solarPV
sector: power

copy_from: windturbine

"""


# File: technodata/{sector}/CommIn.csv
# Copy row(s) where ProcessName={copy_from}, changing ProcessName to {process_name}
# >>> Change solar value to 1
# >>> Change wind value to 0

# File: technodata/{sector}/CommOut.csv
# Copy row(s) where ProcessName={copy_from}, changing ProcessName to {process_name}
# >>> No values changed

# File: technodata/{sector}/ExistingCapacity.csv
# Copy row(s) where ProcessName={copy_from}, changing ProcessName to {process_name}
# >>> No values changed

# File: technodata/{sector}/technodata.csv
# Copy row(s) where ProcessName={copy_from}, changing ProcessName to {process_name}
# >>> Change cap_par to 30
# >>> Change Fuel to 'solar'
