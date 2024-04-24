# starting point: unclear?

"""
ADD A NEW REGION

region_name: R2

copy_from: R1

"""

# File: settings.toml
# Append {region_name} to regions

# For every sector
# File: technodata/{sector}/technodata.csv
# Copy rows where RegionName={copy_from}, changing RegionName to {region_name}
# >>> For sector 'power': MaxCapacityAddition for {windturbine,R2} changed to 5
# >>> For sector 'power': MaxCapacityGrowth for {windturbine,R2} changed to 0.05
# >>> For sector 'power': TotalCapacityLimit for {windturbine,R2} changed to 100

# For every sector
# File: technodata/{sector}/CommIn.csv
# Copy rows where RegionName={copy_from}, changing RegionName to {region_name}
# >>> No values changed

# For every sector
# File: technodata/{sector}/CommOut.csv
# Copy rows where RegionName={copy_from}, changing RegionName to {region_name}
# >>> No values changed

# For every sector
# File: technodata/{sector}/ExistingCapacity.csv
# Copy rows where RegionName={copy_from}, changing RegionName to {region_name}
# >>> No values changed

# File: technodata/Agents.csv (undocumented)
# Copy rows where RegionName={copy_from}, changing RegionName to {region_name}
# >>> No values changed

# File: input/BaseYearImport.csv (undocumented)
# Copy rows where RegionName={copy_from}, changing RegionName to {region_name}
# >>> No values changed

# File: input/BaseYearExport.csv (undocumented)
# Copy rows where RegionName={copy_from}, changing RegionName to {region_name}
# >>> No values changed

# File: input/Projections.csv (undocumented)
# Copy rows where RegionName={copy_from}, changing RegionName to {region_name}
# >>> No values changed

# For all files in technodata/preset (undocumented)
# Copy rows where RegionName={copy_from}, changing RegionName to {region_name}
# >>> No values changed
