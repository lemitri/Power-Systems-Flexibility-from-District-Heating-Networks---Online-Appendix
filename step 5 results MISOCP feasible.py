# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:56:07 2017

@author: lemitri
"""

#%% if infeasible: check IIS

dispatch_feasible.computeIIS()
constraints = dispatch_feasible.model.getConstrs()
variables = dispatch_feasible.model.getVars()

#Print the names of all of the constraints in the IIS set.
for c in constraints:
    if c.IISConstr>0:
        print(c.ConstrName)               

#Print the names of all of the variables in the IIS set.
for v in variables:
    if v.IISLB>0:
        print('lower bound:')
        print(v.VarName)
    if v.IISUB>0:
        print('upper bound:')
        print(v.VarName)

#%% if feasible: variables

P_feasible = {}
for g in elec_station:
    for t in time:
        P_feasible[g,t]=dispatch_feasible.variables.P[g,t].x
for g in elec_storage:
    for t in time:
        P_feasible[g,t]=dispatch_feasible.variables.storage_plus[g,t].x - dispatch_feasible.variables.storage_moins[g,t].x

Q_feasible = {}
for g in heat_station:
    for t in time:
        Q_feasible[g,t]=dispatch_feasible.variables.Q[g,t].x
for g in heat_storage:
    for t in time:
        Q_feasible[g,t]=dispatch_feasible.variables.storage_plus[g,t].x - dispatch_feasible.variables.storage_moins[g,t].x      
        
mf_pipe_feasible = mf_pipe
mf_HES_feasible = mf_HES
mf_HS_feasible = mf_HS
mf_slack_feasible = mf_slack

T_in_feasible = {}
T_out_feasible = {}
T_supply_feasible = {}
T_return_feasible = {}

for p in pipe_supply+pipe_return:
    for t in time_extended[p]:
        T_in_feasible[p,t] = dispatch_feasible.variables.T_in[p,t].x

for p in pipe_supply+pipe_return:
    for t in time:
        T_out_feasible[p,t] = dispatch_feasible.variables.T_out[p,t].x

for t in time:
    for n in node:
        T_supply_feasible[n,t] = dispatch_feasible.variables.T_supply[n,t].x

for t in time:
    for n in node:
        T_return_feasible[n,t] = dispatch_feasible.variables.T_return[n,t].x


time_delay_feasible = {}
for t in time:
    for p in pipe_supply+pipe_return:
        time_delay_feasible[p,t]=dispatch_feasible.variables.tau[p,t].x
        
syst_cost_feasible = dispatch_feasible.model.objval

#%%

#for t in time:
#    for n in node:
#        if T_supply_th[n,t]>-999:
#            print('T supply diff{0}{1}'.format(n,t),T_supply_th[n,t]-T_supply[n,t])
#        if T_return_th[n,t]>-999:
#            print('T return diff{0}{1}'.format(n,t),T_return_th[n,t]-T_return[n,t])
# 
##%%
#
#for p in pipe_supply+pipe_return:
#    plt.plot(range(T),[T_in[p,t] for t in time])
#
##%%
#
#for p in heat_only+heat_pump+CHP:
#    plt.plot(range(T),[Q[p,t] for t in time])
#
##%%
#    
#for p in heat_only+heat_pump+CHP:
#    plt.plot(range(T),[Q_th[p,t] for t in time],ls='--')