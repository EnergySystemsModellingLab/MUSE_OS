"""
ADD A NEW TIMESLICE

timeslice_name: early-morning and late-afternoon
timeslice_number: 7 and 8

copy_from: evening (number=6)

"""

# File: settings.toml
# Append {timeslice_name} to timeslices

# For all files in technodata/preset
# Copy rows for where Timeslice={copy_from.number},
#   changing Timeslice to {timeslice_number}
# >>> No values changed
