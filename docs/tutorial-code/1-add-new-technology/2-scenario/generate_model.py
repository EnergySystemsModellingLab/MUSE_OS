# Starting point: copy model from 1-introduction

"""
ADD PRICE DATA FOR NEW YEAR:

year: 2040
sector: power

copy_from: 2030

"""

# File: technodata/{sector}/technodata.csv
# Copy rows where Time={copy_from}, replacing Time with {year}
# >>> Change 'cap_par' for solarPV,2040 to 30

# File: technodata/{sector}/CommIn.csv
# Copy rows where Time={copy_from}, replacing Time with {year}
# >>> No values changed

# File: technodata/{sector}/CommOut.csv
# Copy rows where Time={copy_from}, replacing Time with {year}
# >>> No values changed
