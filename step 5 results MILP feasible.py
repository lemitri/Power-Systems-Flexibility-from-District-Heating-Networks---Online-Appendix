# -*- coding: utf-8 -*-
"""
Created on Fri Sep  1 09:56:07 2017

@author: lemitri
"""


#%% if infeasible: check IIS

#dispatch_MILP_feasible.computeIIS()
#constraints = dispatch_MILP_feasible.model.getConstrs()
#variables = dispatch_MILP_feasible.model.getVars()
#
##Print the names of all of the constraints in the IIS set.
#for cc in constraints:
#    if cc.IISConstr>0:
#        print(cc.ConstrName)               
#
##Print the names of all of the variables in the IIS set.
#for vv in variables:
#    if vv.IISLB>0:
#        print('lower bound:')
#        print(vv.VarName)
#    if vv.IISUB>0:
#        print('upper bound:')
#        print(vv.VarName)

#%% if feasible: variables

P_MILP_feasible = {}
for g in elec_station:
    for t in time:
        P_MILP_feasible[g,t]=dispatch_MILP_feasible.variables.P[g,t].x
for g in elec_storage:
    for t in time:
        P_MILP_feasible[g,t]=dispatch_MILP_feasible.variables.storage_plus[g,t].x - dispatch_MILP_feasible.variables.storage_moins[g,t].x

Q_MILP_feasible = {}
for g in heat_station:
    for t in time:
        Q_MILP_feasible[g,t]=dispatch_MILP_feasible.variables.Q[g,t].x
for g in heat_storage:
    for t in time:
        Q_MILP_feasible[g,t]=dispatch_MILP_feasible.variables.storage_plus[g,t].x - dispatch_MILP_feasible.variables.storage_moins[g,t].x      
        
mf_pipe_MILP_feasible = mf_pipe_MILP
mf_HES_MILP_feasible = mf_HES_MILP
mf_HS_MILP_feasible = mf_HS_MILP
mf_slack_MILP_feasible = mf_slack_MILP

T_in_MILP_feasible = {}
T_out_MILP_feasible = {}
T_supply_MILP_feasible = {}
T_return_MILP_feasible = {}

for p in pipe_supply+pipe_return:
    for t in time_extended[p]:
        T_in_MILP_feasible[p,t] = dispatch_MILP_feasible.variables.T_in[p,t].x

for p in pipe_supply+pipe_return:
    for t in time:
        T_out_MILP_feasible[p,t] = dispatch_MILP_feasible.variables.T_out[p,t].x

for t in time:
    for n in node:
        T_supply_MILP_feasible[n,t] = dispatch_MILP_feasible.variables.T_supply[n,t].x

for t in time:
    for n in node:
        T_return_MILP_feasible[n,t] = dispatch_MILP_feasible.variables.T_return[n,t].x


time_delay_MILP_feasible = time_delay_MILP
       
syst_cost_MILP_feasible = dispatch_MILP_feasible.model.objval


heat_load_th_MILP_feasible = {}
Q_th_MILP_feasible = {}

for n in node:
    for g in heat_station_node[n]:
        for t in time:
            Q_th_MILP_feasible[g,t]=10**(-6)*water_heat_capacity*Dt*mf_HS_MILP_feasible[g,t]*(dispatch_MILP_feasible.variables.T_supply[n,t].x-dispatch_MILP_feasible.variables.T_return[n,t].x)

for n in node:
    for t in time:
        heat_load_th_MILP_feasible[n,t]=10**(-6)*water_heat_capacity*Dt*mf_HES_MILP_feasible[n,t]*(dispatch_MILP_feasible.variables.T_supply[n,t].x-dispatch_MILP_feasible.variables.T_return[n,t].x)
        
syst_cost_th_MILP_feasible = sum(alpha[g,t]*P_MILP_feasible[g,t] for t in time for g in gen+wind)+sum(alpha[g,t]*Q_th_MILP_feasible[g,t] for t in time for g in heat_only)+sum(alpha[g,t]*(rho_elec[g]*P_MILP_feasible[g,t]+rho_heat[g]*Q_th_MILP_feasible[g,t]) for t in time for g in CHP)


#%%

wind_penetration_MILP_feasible = sum(P_MILP_feasible[w,t] for w in wind for t in time)/(sum(elec_load[n,t] for n in node for t in time)-sum(P_MILP_feasible[h,t] for h in heat_pump for t in time))

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