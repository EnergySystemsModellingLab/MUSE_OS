.. _inputs-commodities:

=====================
Global Commodities
=====================

MUSE handles a configurable number and type of commodities which are primarily used to
represent energy, services, pollutants/emissions. The commodities for the simulation as
a whole are defined in a csv file with the following structure.

.. csv-table:: Global commodities
   :header: commodity, description, commodity_type, unit

   hardcoal, Coal, Energy, PJ
   agires, Agricultural-residues, Energy, PJ

``commodity``
   the internal name used for a commodity inside the model (e.g. "heat" or "elec").
   Any references to commodities in other files must use these names.

``description`` (optional)
   an extended name/description of a commodity (e.g. "Heating" or "Electricity").

``commodity_type``
   defines the type of a commodity:

   The **energy** type includes energy commodities, such as biomass, electricity, gasoline, and hydrogen,
   which are either extracted, transformed from one to another, or used in the energy system.

   The **service** type includes commodities such as space heating or hot water which correspond to selected
   people's needs, and whose fulfillment requires energy uses.

   The **material** type represent non-energy inputs for energy technologies, such as limestone or oxygen.

   The **environmental** type refers to non-energy commodities whichare used to quantify an impact on the environment,
   such as greenhouse gases or CO2. They can be subjected to different types of environmental fees or taxes.

``unit`` (optional)
   is the unit used to represent quantities of the commodity (e.g "PJ").
   This parameter does not need to be included, as it isn't used in the model, but is
   highly recommended for documentation purposes.
   In any case, care should be taken to ensure that units are consistent across all input files.

Additional optional columns
   Users can provide additional columns for extra information about the commodity (e.g. ``heat_rate``).
   These will be ignored by the model, but can be useful for documentation purposes.
