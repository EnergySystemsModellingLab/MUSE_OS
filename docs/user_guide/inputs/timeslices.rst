.. _inputs-legacy-timeslices:

===========
Time-slices
===========

.. note::

    This input file is only for legacy sectors. For anything else, please see :ref:`TOML
    timeslices<user_guide/inputs/toml:Timeslices>`



Time-slices represent a sub-year disaggregation of commodity demand. They are fully
flexible in number and names as to serve the specific representation of the commodity
demand, supply, and supply cost profile in each energy sector.  Each time slice is
independent in terms of number of represent hour, as long as it is meaningful for the
users and their data inputs. 1 is the minimum number of time-slice as this would
correspond to a full year.  The time-slice definition of a sector affects the commodity
price profile and the supply cost profile.

The csv file for the time-slice definition would report the length (in hours) of each
time slice as characteristic to the selected sector to represent diurnal, weekly and
seasonal variation of energy commodities, demand and supply, as shown in the table for
30 time-slices.

.. csv-table:: Time-slices
   :header: AgLevel, SN, Month, Day, Hour, RepresentHours
   :stub-columns: 4

   Hour, 1, Winter, Weekday, Night, 396
   Hour, 2, Winter, Weekday, Morning, 396
   Hour, 3, Winter, Weekday, Afternoon, 264
   Hour, 4, Winter, Weekday, EarlyPeak, 66
   Hour, 5, Winter, Weekday, LatePeak, 66
   Hour, 6, Winter, Weekday, Evening, 396
   Hour, 7, Winter, Weekend, Night, 156
   Hour, 8, Winter, Weekend, Morning, 156
   Hour, 9, Winter, Weekend, Afternoon, 156
   Hour, 10, Winter, Weekend, Evening, 156
   Hour, 11, SpringAutumn, Weekday, Night, 792
   Hour, 12, SpringAutumn, Weekday, Morning, 792
   Hour, 13, SpringAutumn, Weekday, Afternoon, 528
   Hour, 14, SpringAutumn, Weekday, EarlyPeak, 132
   Hour, 15, SpringAutumn, Weekday, LatePeak, 132
   Hour, 16, SpringAutumn, Weekday, Evening, 792
   Hour, 17, SpringAutumn, Weekend, Night, 300
   Hour, 18, SpringAutumn, Weekend, Morning, 300
   Hour, 19, SpringAutumn, Weekend, Afternoon, 300
   Hour, 20, SpringAutumn, Weekend, Evening, 300
   Hour, 21, Summer, Weekday, Night, 396
   Hour, 22, Summer, Weekday, Morning, 396
   Hour, 23, Summer, Weekday, Afternoon, 264
   Hour, 24, Summer, Weekday, EarlyPeak, 66
   Hour, 25, Summer, Weekday, LatePeak, 66
   Hour, 26, Summer, Weekday, Evening, 396
   Hour, 27, Summer, Weekend, Night, 150
   Hour, 28, Summer, Weekend, Morning, 150
   Hour, 29, Summer, Weekend, Afternoon, 150
   Hour, 30, Summer, Weekend, Evening, 150




It reports the aggregation level of the sector time-slices (AgLevel), slice number (SN),
seasonal time slices (Month), weekly time slices (Day), hourly profile (Hour), the
amount of hours associated to each time slice (RepresentHours).
