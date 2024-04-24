"""
ADD AN AGENT

agent_name: A2
- agentshare_new = Agent3
- agentshare_retrofit = Agent4

copy_from: 'A1'

"""

# File: technodata/Agents.csv
# Copy rows where Name={copy_from}, changing Name to {agent_name}
# Rename AgentShare for new rows to {agentshare_new} (where Type=New)
#   and {agentshare_retrofit} (where Type=Retrofit)
# >>> Change Objective2 to 'EAC'
# >>> Change ObjData1 and ObjData2 to 0.5
# >>> Change DecisionMethod to 'weighted_sum'

# For every sector
# File: technodata/{sector}/technodata.csv
# Create column {agentshare_retrofit} as copy of {copy_from.agentshare_retrofit}
# >>> No values changed
