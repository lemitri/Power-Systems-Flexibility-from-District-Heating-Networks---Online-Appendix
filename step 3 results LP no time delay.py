# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:56:07 2017

@author: lemitri
"""


#%% if infeasible: check IIS

#dispatch.computeIIS()
#constraints = dispatch.model.getConstrs()
#variables = dispatch.model.getVars()
#
##Print the names of all of the constraints in the IIS set.
#for c in constraints:
#    if c.IISConstr>0:
#        print(c.ConstrName)               
#
##Print the names of all of the variables in the IIS set.
#for v in variables:
#    if v.IISLB>0:
#        print('lower bound:')
#        print(v.VarName)
#    if v.IISUB>0:
#        print('upper bound:')
#        print(v.VarName)

#%% if feasible: variables

syst_cost_LP = dispatch_LP.model.objval

P_LP = {}
for g in elec_station:
    for t in time:
        P_LP[g,t]=dispatch_LP.variables.P[g,t].x
for g in elec_storage:
    for t in time:
        P_LP[g,t]=dispatch_LP.variables.storage_plus[g,t].x - dispatch_LP.variables.storage_moins[g,t].x

Q_LP = {}
for g in heat_station:
    for t in time:
        Q_LP[g,t]=dispatch_LP.variables.Q[g,t].x
for g in heat_storage:
    for t in time:
        Q_LP[g,t]=dispatch_LP.variables.storage_plus[g,t].x - dispatch_LP.variables.storage_moins[g,t].x      

#%%

wind_utilization_LP = sum(P_LP[w,t] for w in wind for t in time)/(sum(wind_scenario[w,t]*elec_maxprod[w] for w in wind for t in time))       
#wind_utilization_MILP_feasible = sum(P_MILP_feasible[w,t] for w in wind for t in time)/(sum(wind_scenario[w,t]*elec_maxprod[w] for w in wind for t in time))
#
#diff_wind_utilization = sum(P_MILP_feasible[w,t] for w in wind for t in time)-sum(P_LP[w,t] for w in wind for t in time)
#diff_percentage_wind_utilization = wind_utilization_MILP_feasible - wind_utilization_LP
#
#print('diff_percentage_wind_utilization',diff_percentage_wind_utilization) 
#print('diff_wind_utilization',diff_wind_utilization)   
#  
 
#%%

#print('Cost MILP',syst_cost_MILP)
#print('Cost MILP feasible',syst_cost_MILP_feasible)
print('Cost LP',syst_cost_LP)
#print('Cost % difference',(syst_cost_LP-syst_cost_MILP)/syst_cost_LP*100)
#print('Cost % difference feasible',(syst_cost_LP-syst_cost_MILP_feasible)/syst_cost_LP*100)

