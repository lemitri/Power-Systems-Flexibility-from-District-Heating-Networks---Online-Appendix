#%%
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 11 13:43:46 2016

@author: lemitri
"""

# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 10:31:11 2016

@author: lemitri
"""


import os
import pandas as pd
import scipy.stats as sp
#import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb
sb.set_style('ticks')

import gurobipy as gb
import itertools as it

import numpy as np
import datetime as dtt
import time as tt

#os.chdir("C:/Users/lemitri/Documents/phd/Combined Heat and Power Flow/python/data")

    
#%% building the STOCHSTICA MPEC optimization problem (GUROBI) = heat market clearing

class expando(object):
    '''
    
    
        A small class which can have attributes set
    '''
    pass

class integrated_dispatch_MILP:
    def __init__(self):
        self.data = expando()
        self.variables = expando()
        self.constraints = expando()
        self._load_data()
        self._build_model()

    
    def optimize(self):
        self.model.optimize()

    def computeIIS(self):
        self.model.computeIIS()
    
    def _load_data(self):
        
        #indexes
        self.data.time = time
        self.data.time_list=time_list
        self.data.node=node
        self.data.line=line
        self.data.pipe_supply=pipe_supply
        self.data.pipe_return=pipe_return
        self.data.gen=gen
        self.data.heat_storage=heat_storage
        self.data.elec_storage=elec_storage
        self.data.heat_pump =heat_pump
        self.data.wind=wind
        self.data.heat_only = heat_only
        self.data.CHP_sorted = CHP_sorted
        self.data.CHP = CHP
        self.data.producers=producers
        self.data.heat_station=heat_station
        self.data.elec_station=elec_station
        #self.data.heat_exchanger_station = heat_exchanger_station
        
        #epsilon
        self.data.epsilon = 0.000001
        
        #producers sorted per node
        self.data.producers_node = producers_node
        self.data.heat_station_node=heat_station_node
        self.data.elec_station_node=elec_station_node
        self.data.CHP_node = CHP_node
        self.data.CHP_sorted_node= CHP_sorted_node
        self.data.heat_pump_node = heat_pump_node
        self.data.heat_only_node = heat_only_node
        self.data.heat_storage_node= heat_storage_node
        self.data.gen_node = gen_node
        self.data.wind_node= wind_node
        self.data.elec_storage_node = elec_storage_node
        #self.data.heat_exchanger_station_node = heat_exchanger_station_node
        
        # connexions between nodes
        self.data.pipe_supply_connexion= pipe_supply_connexion
        self.data.pipe_supply_start = pipe_supply_start
        self.data.pipe_supply_end = pipe_supply_end
        self.data.pipe_return_connexion= pipe_return_connexion
        self.data.pipe_return_start = pipe_return_start
        self.data.pipe_return_end = pipe_return_end
        self.data.line_connexion= line_connexion
        self.data.line_start = line_start
        self.data.line_end = line_end
        
        # heat network parameters
        self.data.pressure_loss_coeff = pressure_loss_coeff
        self.data.radius_pipe = radius_pipe
        self.data.water_density = water_density
        self.data.water_thermal_loss_coeff = water_thermal_loss_coeff
        self.data.water_heat_capacity = water_heat_capacity
        self.data.length_pipe = length_pipe
        
        #pressure bounds
        self.data.pr_return_max = pr_return_max
        self.data.pr_return_min = pr_return_min
        self.data.pr_supply_max = pr_supply_max
        self.data.pr_supply_min = pr_supply_min
        self.data.pr_diff_min = pr_supply_min
        self.data.pr_return_slice = pr_return_slice
        self.data.pr_supply_slice = pr_supply_slice

        # temperature bounds
        self.data.T_in_max = T_in_max
        self.data.T_in_min = T_in_min
        self.data.T_out_max = T_out_max
        self.data.T_out_min = T_out_min
        self.data.T_return_max = T_return_max
        self.data.T_return_min = T_return_min
        self.data.T_supply_max = T_supply_max
        self.data.T_supply_min = T_supply_min
        
        #mass flow bounds        
        self.data.mf_pipe_max = mf_pipe_max
        self.data.mf_pipe_min = mf_pipe_min
        self.data.mf_HS_max = mf_HS_max
        self.data.mf_HS_min = mf_HS_min
        self.data.mf_HES_max = mf_HES_max
        self.data.mf_HES_min = mf_HES_min       
        
        # LOADS
        self.data.heat_load = heat_load
        self.data.elec_load = elec_load
        
        # Heat station parameters
        self.data.CHP_maxprod = CHP_maxprod
        self.data.heat_maxprod = heat_maxprod
        self.data.rho_elec = rho_elec
        self.data.rho_heat = rho_heat
        self.data.r_min = r_min
        self.data.storage_rho_plus= storage_rho_plus
        self.data.storage_rho_moins= storage_rho_moins
        self.data.storage_maxcapacity= storage_maxcapacity
        self.data.storage_loss= storage_loss
        self.data.storage_init= storage_init
        self.data.COP = COP 
        
        # Elec station parameters
        self.data.elec_maxprod = elec_maxprod
        self.data.wind_scenario = wind_scenario

        # Cost parameters
        self.data.alpha = alpha    
        
        # time delay
        self.data.Dt = Dt
        self.data.time_delay_max = time_delay_max
        self.data.time_delay_range = time_delay_range
        self.data.mf_pipe_init = mf_pipe_init
        self.data.T_in_init = T_in_init
        
        self.data.time_init = time_init
        self.data.time_extended = time_extended
        
        self.data.tau_new_min = tau_new_min
        self.data.tau_new_max = tau_new_max        
        
        self.data.big_M = 1+max([self.data.mf_pipe_max[p]*self.data.Dt*(self.data.time_delay_max[p]+1)/(np.pi*self.data.radius_pipe[p]*self.data.radius_pipe[p]*self.data.water_density) - self.data.length_pipe[p] for p in pipe_supply+pipe_return])

        #elec transmission system
        self.data.B = B
        self.data.line_maxflow = line_maxflow
   
    def _build_model(self):
        
        self.model = gb.Model()
        self._build_variables()
        self._build_objective()
        self._build_constraints()
    
    def _build_variables(self):
        
        #indexes shortcuts 
        time = self.data.time
        time_init = self.data.time_init
        time_extended = self.data.time_extended
        node=self.data.node
        line=self.data.line
        pipe_supply=self.data.pipe_supply
        pipe_return=self.data.pipe_return
        gen=self.data.gen
        heat_storage=self.data.heat_storage
        elec_storage=self.data.elec_storage
        heat_pump=self.data.heat_pump
        wind=self.data.wind
        heat_only=self.data.heat_only
        CHP_sorted=self.data.CHP_sorted
        CHP=self.data.CHP
        producers=self.data.producers
        heat_station=self.data.heat_station
        elec_station=self.data.elec_station
        #heat_exchanger_station=self.data.heat_exchanger_station       
        m = self.model
        
        #heat market optimization variables

        self.variables.Q = {} #heat production from CHPs and HO units (first satge)
        for t in time:
            for h in heat_station:
                self.variables.Q[h,t] = m.addVar(lb=0,ub=self.data.heat_maxprod[h],name='Q({0},{1})'.format(h,t))

        self.variables.storage_plus = {} #heat storage: heat discharged (first stage)
        for t in time:
            for h in heat_storage+elec_storage:
                self.variables.storage_plus[h,t] = m.addVar(lb=0,ub=self.data.storage_maxprod[h],name='storage plus({0},{1})'.format(h,t))
                    
        self.variables.storage_moins = {} #heat storage: heat charged (first stage)
        for t in time:
            for h in heat_storage+elec_storage:
                self.variables.storage_moins[h,t] = m.addVar(lb=0,ub=self.data.storage_maxprod[h],name='storage moins({0},{1})'.format(h,t))
                    

        self.variables.storage_energy = {} #heat stored in heat storage h at end of time period t
        for t in time:
            for h in heat_storage+elec_storage:
                self.variables.storage_energy[h,t] = m.addVar(lb=0,ub=self.data.storage_maxcapacity[h],name='storage energy({0},{1})'.format(h,t))                

        #electricity market optimization variables : primal variables
        
        self.variables.P = {} # electricity production from electricity generators, CHPs and wind producers
        for t in time:
            for g in CHP+gen+wind:
                self.variables.P[g,t] = m.addVar(lb=0,ub=self.data.elec_maxprod[g],name='P({0},{1})'.format(g,t)) # dispatch of electricity generators
            
            for g in heat_pump:
                self.variables.P[g,t] = m.addVar(lb=-gb.GRB.INFINITY,ub=0,name='P({0},{1})'.format(g,t)) # elec consumption of HP
                
        #masse flow rates!!!
                
        self.variables.mf_pipe = {}
        for p in pipe_supply+pipe_return:
            for t in self.data.time_extended[p]:
                self.variables.mf_pipe[p,t] = m.addVar(lb=self.data.mf_pipe_min[p],ub=self.data.mf_pipe_max[p],name='mf pipe({0},{1})'.format(p,t))
             
        self.variables.mf_HES = {}
        for t in time:
            for p in node:
                self.variables.mf_HES[p,t] = m.addVar(lb=self.data.mf_HES_min[p],ub=self.data.mf_HES_max[p],name='mf HES({0},{1})'.format(p,t))

        self.variables.mf_HS = {}
        for t in time:
            for p in heat_station+heat_storage:
                self.variables.mf_HS[p,t] = m.addVar(lb=self.data.mf_HS_min[p],ub=self.data.mf_HS_max[p],name='mf HS({0},{1})'.format(p,t)) 
        
        #temperatures
        self.variables.T_in = {}
        for p in pipe_supply+pipe_return:
            for t in time_extended[p]:
                self.variables.T_in[p,t] = m.addVar(lb=self.data.T_in_min[p,t],ub=self.data.T_in_max[p,t],name='temp in({0},{1})'.format(p,t))

        self.variables.T_out = {}
        for p in pipe_supply+pipe_return:
            for t in time:
                self.variables.T_out[p,t] = m.addVar(lb=self.data.T_out_min[p,t],ub=self.data.T_out_max[p,t],name='temp out({0},{1})'.format(p,t))

        self.variables.T_supply = {}
        for t in time:
            for n in node:
                self.variables.T_supply[n,t] = m.addVar(lb=self.data.T_supply_min[n,t],ub=self.data.T_supply_max[n,t],name='temp supply({0},{1})'.format(p,t))        

        self.variables.T_return = {}
        for t in time:
            for n in node:
                self.variables.T_return[n,t] = m.addVar(lb=self.data.T_return_min[n,t],ub=self.data.T_return_max[n,t],name='temp return({0},{1})'.format(p,t))

        # pressure variables
                
        self.variables.pr_supply = {}
        for t in time:
            for n in node:
                self.variables.pr_supply[n,t] = m.addVar(lb=self.data.pr_supply_min[n],ub=self.data.pr_supply_max[n],name='pr supply({0},{1})'.format(n,t))        

        self.variables.pr_return = {}
        for t in time:
            for n in node:
                self.variables.pr_return[n,t] = m.addVar(lb=self.data.pr_return_min[n],ub=self.data.pr_return_max[n],name='pr return({0},{1})'.format(n,t))
        
        # time delays in pipes (discrete) 

        self.variables.tau = {}
        for t in time:
            for p in pipe_supply+pipe_return:
                self.variables.tau[p,t] = m.addVar(lb=0,ub=self.data.time_delay_max[p],name='tau({0},{1})'.format(p,t))

        self.variables.u = {}
        for t in time:
            for p in pipe_supply+pipe_return:
                for n in range(self.data.time_delay_max[p]+1):
                    self.variables.u[p,n,t] = m.addVar(vtype=gb.GRB.BINARY,name='u({0},{1},{2})'.format(p,n,t))
                    
        self.variables.T_in_new = {}
        for p in pipe_supply+pipe_return:
            for t in time:
                for n in range(self.data.time_delay_max[p]+1):
                    self.variables.T_in_new[p,n,t] = m.addVar(lb=-gb.GRB.INFINITY,name='temp in new({0},{1},{2})'.format(p,n,t))

        self.variables.v = {}
        for t in time:
            for p in pipe_supply+pipe_return:
                for n in range(self.data.time_delay_max[p]+1):
                    self.variables.v[p,n,t] = m.addVar(vtype=gb.GRB.BINARY,name='v({0},{1},{2})'.format(p,n,t))


        # electricity transmission system variables 

        self.variables.node_angle = {}
        for t in time:
            for n in node:
                    self.variables.node_angle[n,t] = m.addVar(lb=-gb.GRB.INFINITY,name='node angle({0},{1})'.format(n,t))

        self.variables.flow_line = {}
        for t in time:
            for l in line:
                self.variables.flow_line[l,t] = m.addVar(lb=-self.data.line_maxflow[l],ub=self.data.line_maxflow[l],name='flow line({0},{1})'.format(l,t))                    


#         #temp mixing relaxation: auxiliary variables
#
#        self.variables.w = {}
#        for t in time:
#            for p in pipe_supply+pipe_return:
#                self.variables.w[p,t] = m.addVar(lb=-gb.GRB.INFINITY,name='w({0},{1})'.format(p,t))

        # SLACK VARIABLES (FEASIBILITY)

        self.variables.mf_supply_slack_plus = {}
        for t in time:
            for n in node:
                self.variables.mf_supply_slack_plus[n,t] = m.addVar(lb=0,ub=2000,name='mf slack supply plus({0},{1})'.format(n,t))

        self.variables.mf_supply_slack_moins = {}
        for t in time:
            for n in node:
                self.variables.mf_supply_slack_moins[n,t] = m.addVar(lb=0,ub=2000,name='mf slack supply moins({0},{1})'.format(n,t))

        self.variables.mf_return_slack_plus = {}
        for t in time:
            for n in node:
                self.variables.mf_return_slack_plus[n,t] = m.addVar(lb=0,ub=2000,name='mf slack return plus({0},{1})'.format(n,t))

        self.variables.mf_return_slack_moins = {}
        for t in time:
            for n in node:
                self.variables.mf_return_slack_moins[n,t] = m.addVar(lb=0,ub=2000,name='mf slack return moins({0},{1})'.format(n,t))
                
        m.update()
    
    def _build_objective(self): # building the objective function for the heat maret clearing

        #indexes shortcuts 
        time = self.data.time
        time_init = self.data.time_init
        time_extended = self.data.time_extended
        node=self.data.node
        line=self.data.line
        pipe_supply=self.data.pipe_supply
        pipe_return=self.data.pipe_return
        gen=self.data.gen
        heat_storage=self.data.heat_storage
        elec_storage=self.data.elec_storage
        heat_pump=self.data.heat_pump
        wind=self.data.wind
        heat_only=self.data.heat_only
        CHP_sorted=self.data.CHP_sorted
        CHP=self.data.CHP
        producers=self.data.producers
        heat_station=self.data.heat_station
        elec_station=self.data.elec_station
        #heat_exchanger_station=self.data.heat_exchanger_station       
        m = self.model     
              
        m.setObjective(0.00001*gb.quicksum(self.variables.tau[p,t] for t in time for p in pipe_supply+pipe_return)+1000*gb.quicksum(self.variables.mf_supply_slack_moins[n,t]+self.variables.mf_supply_slack_plus[n,t]+self.variables.mf_return_slack_moins[n,t]+self.variables.mf_return_slack_plus[n,t] for n in node for t in time)+gb.quicksum(self.data.alpha[g,t]*self.variables.P[g,t] for t in time for g in gen+wind)+gb.quicksum(self.data.alpha[g,t]*self.variables.Q[g,t] for t in time for g in heat_only)+gb.quicksum(self.data.alpha[g,t]*(self.data.rho_elec[g]*self.variables.P[g,t]+self.data.rho_heat[g]*self.variables.Q[g,t]) for t in time for g in CHP),   
            gb.GRB.MINIMIZE)
            
        
    def _build_constraints(self):

        #indexes shortcuts 
        time = self.data.time
        time_init = self.data.time_init
        time_extended = self.data.time_extended
        node=self.data.node
        line=self.data.line
        pipe_supply=self.data.pipe_supply
        pipe_return=self.data.pipe_return
        gen=self.data.gen
        heat_storage=self.data.heat_storage
        elec_storage=self.data.elec_storage
        heat_pump=self.data.heat_pump
        wind=self.data.wind
        heat_only=self.data.heat_only
        CHP_sorted=self.data.CHP_sorted
        CHP=self.data.CHP
        producers=self.data.producers
        heat_station=self.data.heat_station
        elec_station=self.data.elec_station
        #heat_exchanger_station=self.data.heat_exchanger_station       
        m = self.model
        
        # HES equations   
    
#        self.constraints.heat_load = {}
#        
#        for t in time:
#            for n in node:
#                self.constraints.heat_load[n,t] = m.addConstr(
#                        10**(-6)*self.data.water_heat_capacity*self.data.Dt*self.variables.mf_HES[n,t]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t]),
#                        gb.GRB.EQUAL,
#                        self.data.heat_load[n,t])

        self.constraints.heat_load_1 = {}
        
        for t in time:
            for n in node:
                self.constraints.heat_load_1[n,t] = m.addConstr(
                        self.data.heat_load[n,t],
                        gb.GRB.GREATER_EQUAL,
                        10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HES_max[n]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HES[n,t]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])-self.data.mf_HES_max[n]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])),name='heat load 1({0},{1})'.format(n,t))

        self.constraints.heat_load_2 = {}
        
        for t in time:
            for n in node:
                self.constraints.heat_load_2[n,t] = m.addConstr(
                        self.data.heat_load[n,t],
                        gb.GRB.GREATER_EQUAL,
                        10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HES_min[n]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HES[n,t]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])-self.data.mf_HES_min[n]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])),name='heat load 2({0},{1})'.format(n,t))

        self.constraints.heat_load_3 = {}
        
        for t in time:
            for n in node:
                self.constraints.heat_load_3[n,t] = m.addConstr(
                        self.data.heat_load[n,t],
                        gb.GRB.LESS_EQUAL,
                        10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HES_max[n]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HES[n,t]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])-self.data.mf_HES_max[n]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])),name='heat load 3({0},{1})'.format(n,t))

        self.constraints.heat_load_4 = {}
        
        for t in time:
            for n in node:
                self.constraints.heat_load_4[n,t] = m.addConstr(
                        self.data.heat_load[n,t],
                        gb.GRB.LESS_EQUAL,
                        10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HES_min[n]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HES[n,t]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])-self.data.mf_HES_min[n]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])),name='heat load 4({0},{1})'.format(n,t))

                                                                        
        self.constraints.pressure_diff = {}
        for t in time:
            for n in node:
                self.constraints.pressure_diff[n,t] = m.addConstr(
                        self.variables.pr_supply[n,t]-self.variables.pr_return[n,t],
                        gb.GRB.GREATER_EQUAL,
                        self.data.pr_diff_min[n],name='pressure diff({0},{1})'.format(n,t))
        
        # HS equations  
              
#        self.constraints.heat_station = {}
#        
#        for t in time:
#            for n in node:
#                
#                for h in self.data.heat_station_node[n]:
#                    self.constraints.heat_station[h,t] = m.addConstr(
#                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*self.variables.mf_HS[h,t]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t]),
#                            gb.GRB.EQUAL,
#                            self.variables.Q[h,t])
#                            
#                for h in self.data.heat_storage_node[n]:
#                    self.constraints.heat_station[h,t] = m.addConstr(
#                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*self.variables.mf_HS[h,t]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t]),
#                            gb.GRB.EQUAL,
#                            self.variables.storage_plus[h,t]-self.variables.storage_moins[h,t])                    


        self.constraints.heat_station_1 = {}
        
        for t in time:
            for n in node:
                for h in self.data.heat_station_node[n]:
                    
                    self.constraints.heat_station_1[h,t] = m.addConstr(
                            self.variables.Q[h,t],
                            gb.GRB.GREATER_EQUAL,
                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HS_max[h]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HS[h,t]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])-self.data.mf_HS_max[h]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])),name='heat station 1({0},{1})'.format(h,t))

                for h in self.data.heat_storage_node[n]:
                    
                    self.constraints.heat_station_1[h,t] = m.addConstr(
                            self.variables.storage_plus[h,t]-self.variables.storage_moins[h,t],
                            gb.GRB.GREATER_EQUAL,
                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HS_max[h]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HS[h,t]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])-self.data.mf_HS_max[h]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])),name='heat station 1({0},{1})'.format(h,t))

                    
        self.constraints.heat_station_2 = {}
        
        for t in time:
            for n in node:
                for h in self.data.heat_station_node[n]:
                    
                    self.constraints.heat_station_2[h,t] = m.addConstr(
                            self.variables.Q[h,t],
                            gb.GRB.GREATER_EQUAL,
                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HS_min[h]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HS[h,t]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])-self.data.mf_HS_min[h]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])),name='heat station 2({0},{1})'.format(h,t))

                for h in self.data.heat_storage_node[n]:

                    self.constraints.heat_station_2[h,t] = m.addConstr(
                            self.variables.storage_plus[h,t]-self.variables.storage_moins[h,t],
                            gb.GRB.GREATER_EQUAL,
                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HS_min[h]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HS[h,t]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])-self.data.mf_HS_min[h]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])),name='heat station 2({0},{1})'.format(h,t))                    
                    
                    
        self.constraints.heat_station_3 = {}
        
        for t in time:
            for n in node:
                for h in self.data.heat_station_node[n]:
                    
                    self.constraints.heat_station_3[h,t] = m.addConstr(
                            self.variables.Q[h,t],
                            gb.GRB.LESS_EQUAL,
                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HS_max[h]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HS[h,t]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])-self.data.mf_HS_max[h]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])),name='heat station 3({0},{1})'.format(h,t))

                for h in self.data.heat_storage_node[n]:
                    
                    self.constraints.heat_station_3[h,t] = m.addConstr(
                            self.variables.storage_plus[h,t]-self.variables.storage_moins[h,t],
                            gb.GRB.LESS_EQUAL,
                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HS_max[h]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HS[h,t]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])-self.data.mf_HS_max[h]*(self.data.T_supply_min[n,t]-self.data.T_return_max[n,t])),name='heat station 3({0},{1})'.format(h,t))
                                        
                   
        self.constraints.heat_station_4 = {}
        
        for t in time:
            for n in node:
                for h in self.data.heat_station_node[n]:
                    
                    self.constraints.heat_station_4[h,t] = m.addConstr(
                            self.variables.Q[h,t],
                            gb.GRB.LESS_EQUAL,
                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HS_min[h]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HS[h,t]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])-self.data.mf_HS_min[h]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])),name='heat station 4({0},{1})'.format(h,t))
    
                for h in self.data.heat_storage_node[n]:
                    
                    self.constraints.heat_station_4[h,t] = m.addConstr(
                            self.variables.storage_plus[h,t]-self.variables.storage_moins[h,t],
                            gb.GRB.LESS_EQUAL,
                            10**(-6)*self.data.water_heat_capacity*self.data.Dt*(self.data.mf_HS_min[h]*(self.variables.T_supply[n,t]-self.variables.T_return[n,t])+self.variables.mf_HS[h,t]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])-self.data.mf_HS_min[h]*(self.data.T_supply_max[n,t]-self.data.T_return_min[n,t])),name='heat station 4({0},{1})'.format(h,t))                    


        ##1) CHPs
                    
        self.constraints.CHP_maxprod = {} 
        self.constraints.CHP_ratio = {}
        
        for t in time:
                for h in self.data.CHP_sorted['ex']:
                    
                    self.constraints.CHP_maxprod[h,t] = m.addConstr(
                        self.data.rho_heat[h]*self.variables.Q[h,t]+self.data.rho_elec[h]*self.variables.P[h,t],
                        gb.GRB.LESS_EQUAL,
                        self.data.CHP_maxprod[h],name='CHP maxprod({0},{1})'.format(h,t))
                    
                    self.constraints.CHP_ratio[h,t] = m.addConstr(
                        self.variables.P[h,t],
                        gb.GRB.GREATER_EQUAL,
                        self.data.r_min[h]*self.variables.Q[h,t],name='CHP ratio({0},{1})'.format(h,t))

                for h in self.data.CHP_sorted['bp']:
                    
                    self.constraints.CHP_ratio[h,t] = m.addConstr(
                        self.variables.P[h,t],
                        gb.GRB.EQUAL,
                        self.data.r_min[h]*self.variables.Q[h,t],name='CHP ratio({0},{1})'.format(h,t)) 
        
        ##2) storage 
               
        self.constraints.storage_update={}
        
        for (t1,t2) in zip(time[:-1],time[1:]):
            for h in heat_storage+elec_storage:
                self.constraints.storage_update[h,t2]=m.addConstr(
                    self.variables.storage_energy[h,t2],
                    gb.GRB.EQUAL,
                    self.variables.storage_energy[h,t1]-self.data.storage_rho_plus[h]*self.variables.storage_plus[h,t2]+self.data.storage_rho_moins[h]*self.variables.storage_moins[h,t2]-self.data.storage_loss[h],name='storage update({0},{1})'.format(h,t2))


        self.constraints.storage_init={}
        
        for h in heat_storage+elec_storage:
            self.constraints.storage_init[h]=m.addConstr(
                self.variables.storage_energy[h,time[0]],
                gb.GRB.EQUAL,
                self.data.storage_init[h]-self.data.storage_rho_plus[h]*self.variables.storage_plus[h,time[0]]+self.data.storage_rho_moins[h]*self.variables.storage_moins[h,time[0]]-self.data.storage_loss[h],name='storage init({0})'.format(h))

        self.constraints.storage_final={}
        
        for h in heat_storage+elec_storage:
            self.constraints.storage_final[h]=m.addConstr(
                self.variables.storage_energy[h,time[-1]],
                gb.GRB.GREATER_EQUAL,
                self.data.storage_init[h],name='storage final({0})'.format(h))
        
        #3) heat pumps
        
        self.constraints.heat_pump_COP = {}
        
        for t in time:
                for h in heat_pump:                    
                    self.constraints.heat_pump_COP[h,t] = m.addConstr(
                        -self.data.COP[h]*self.variables.P[h,t],
                        gb.GRB.EQUAL,
                        self.variables.Q[h,t],name='heat pump COP({0},{1})'.format(h,t)) 
                            
        #continuity of mass flow at nodes
        
        self.constraints.mass_flow_continuity_supply = {}
        self.constraints.mass_flow_continuity_return = {}
        
        for t in time:
                for n in node: 
                    
                    self.constraints.mass_flow_continuity_supply[n,t] = m.addConstr(
                        gb.quicksum(self.variables.mf_HS[h,t] for h in self.data.heat_station_node[n])+gb.quicksum(self.variables.mf_HS[h,t] for h in self.data.heat_storage_node[n])+gb.quicksum(self.variables.mf_pipe[p,t] for p in self.data.pipe_supply_end[n])-gb.quicksum(self.variables.mf_pipe[p,t] for p in self.data.pipe_supply_start[n])+(self.variables.mf_supply_slack_plus[n,t]-self.variables.mf_supply_slack_moins[n,t]),
                        gb.GRB.EQUAL,
                        self.variables.mf_HES[n,t],name='mass flow continuity supply({0},{1})'.format(n,t))         
                  
                    self.constraints.mass_flow_continuity_return[n,t] = m.addConstr(
                        self.variables.mf_HES[n,t]+gb.quicksum(self.variables.mf_pipe[p,t] for p in self.data.pipe_return_end[n])-gb.quicksum(self.variables.mf_pipe[p,t] for p in self.data.pipe_return_start[n])+(self.variables.mf_return_slack_plus[n,t]-self.variables.mf_return_slack_moins[n,t]),
                        gb.GRB.EQUAL,
                        gb.quicksum(self.variables.mf_HS[h,t] for h in self.data.heat_station_node[n])+gb.quicksum(self.variables.mf_HS[h,t] for h in self.data.heat_storage_node[n]),name='mass flow continuity return({0},{1})'.format(n,t))
                        
        # temperature mixing at nodes

        #SIMPLIFIED VERSION (ONLY 1 pipe arriving!!!!!!!!!!!!!!!)


        self.constraints.T_mixing_supply = {}
        self.constraints.T_mixing_return = {}
        
        for t in time:
            for n in node: 
                for p in self.data.pipe_supply_end[n]:                    
                    self.constraints.T_mixing_supply[p,t] = m.addConstr(
                        self.variables.T_supply[n,t],
                        gb.GRB.EQUAL,
                        self.variables.T_out[p,t],name='Temp mixing supply simplified ({0},{1},{2})'.format(n,p,t))         
                 
                for p in self.data.pipe_return_end[n]:
                    self.constraints.T_mixing_return[p,t] = m.addConstr(
                        self.variables.T_return[n,t],
                        gb.GRB.EQUAL,
                        self.variables.T_out[p,t],name='Temp mixing return simplified ({0},{1},{2})'.format(n,p,t))         


##        self.constraints.T_mixing_supply = {}
##        self.constraints.T_mixing_return = {}
##        
##        for t in time:
##                for n in node: 
##                    
##                    self.constraints.T_mixing_supply[n,t] = m.addConstr(
##                        self.variables.T_supply[n,t]*gb.quicksum(self.variables.mf_pipe[p,t] for p in self.data.pipe_supply_end[n]),
##                        gb.GRB.EQUAL,
##                        gb.quicksum(self.variables.T_out[p,t]*self.variables.mf_pipe[p,t] for p in self.data.pipe_supply_end[n]))         
##                    
##                    self.constraints.T_mixing_return[n,t] = m.addConstr(
##                        self.variables.T_return[n,t]*gb.quicksum(self.variables.mf_pipe[p,t] for p in self.data.pipe_return_end[n]),
##                        gb.GRB.EQUAL,
##                        gb.quicksum(self.variables.T_out[p,t]*self.variables.mf_pipe[p,t] for p in self.data.pipe_return_end[n]))         
#
#
#        self.constraints.T_mixing_supply_0 = {}
#        self.constraints.T_mixing_return_0 = {}
#        
#        for t in time:
#                for n in node: 
#                    
#                    self.constraints.T_mixing_supply_0[n,t] = m.addConstr(
#                        gb.quicksum(self.variables.w[p,t] for p in self.data.pipe_supply_end[n]),
#                        gb.GRB.EQUAL,
#                        0,name='T mixing supply 0({0},{1})'.format(n,t))         
#                    
#                    self.constraints.T_mixing_return_0[n,t] = m.addConstr(
#                        gb.quicksum(self.variables.w[p,t] for p in self.data.pipe_return_end[n]),
#                        gb.GRB.EQUAL,
#                        0,name='T mixing return 0({0},{1})'.format(n,t))         
#
##        self.constraints.T_mixing_supply_1 = {}
##        self.constraints.T_mixing_return_1 = {}
##        
##        for t in time:
##                for n in node: 
##                    for p in pipe_supply_end[n]:
##                        self.constraints.T_mixing_supply_1[p,t] = m.addConstr(
##                            self.variables.w[p,t],
##                            gb.GRB.EQUAL,
##                            self.variables.mf_pipe[p,t]*(self.variables.T_supply[n,t]-self.variables.T_out[p,t]))         
##
##                    for p in pipe_return_end[n]:                        
##                        self.constraints.T_mixing_return_1[n,t] = m.addConstr(
##                            self.variables.w[p,t],
##                            gb.GRB.EQUAL,
##                            self.variables.mf_pipe[p,t]*(self.variables.T_return[n,t]-self.variables.T_out[p,t])) 
#
#        self.constraints.T_mixing_supply_1 = {}
#        self.constraints.T_mixing_return_1 = {}
#        
#        for t in time:
#                for n in node: 
#                    for p in self.data.pipe_supply_end[n]:
#                        self.constraints.T_mixing_supply_1[p,t] = m.addConstr(
#                            self.variables.w[p,t],
#                            gb.GRB.GREATER_EQUAL,
#                            self.data.mf_pipe_max[p]*(self.variables.T_supply[n,t]-self.variables.T_out[p,t])+self.variables.mf_pipe[p,t]*(self.data.T_supply_max[n,t]-self.data.T_out_min[p,t])-self.data.mf_pipe_max[p]*(self.data.T_supply_max[n,t]-self.data.T_out_min[p,t]),name='T mixing supply 1({0},{1})'.format(p,t))         
#
#                    for p in self.data.pipe_return_end[n]:                        
#                        self.constraints.T_mixing_return_1[p,t] = m.addConstr(
#                            self.variables.w[p,t],
#                            gb.GRB.GREATER_EQUAL,
#                            self.data.mf_pipe_max[p]*(self.variables.T_return[n,t]-self.variables.T_out[p,t])+self.variables.mf_pipe[p,t]*(self.data.T_return_max[n,t]-self.data.T_out_min[p,t])-self.data.mf_pipe_max[p]*(self.data.T_return_max[n,t]-self.data.T_out_min[p,t]),name='T mixing return 1({0},{1})'.format(p,t))         
#
#        self.constraints.T_mixing_supply_2 = {}
#        self.constraints.T_mixing_return_2 = {}
#        
#        for t in time:
#                for n in node: 
#                    for p in self.data.pipe_supply_end[n]:
#                        self.constraints.T_mixing_supply_2[p,t] = m.addConstr(
#                            self.variables.w[p,t],
#                            gb.GRB.GREATER_EQUAL,
#                            self.data.mf_pipe_min[p]*(self.variables.T_supply[n,t]-self.variables.T_out[p,t])+self.variables.mf_pipe[p,t]*(self.data.T_supply_min[n,t]-self.data.T_out_max[p,t])-self.data.mf_pipe_min[p]*(self.data.T_supply_min[n,t]-self.data.T_out_max[p,t]),name='T mixing supply 2({0},{1})'.format(p,t))         
#
#                    for p in self.data.pipe_return_end[n]:                        
#                        self.constraints.T_mixing_return_2[p,t] = m.addConstr(
#                            self.variables.w[p,t],
#                            gb.GRB.GREATER_EQUAL,
#                            self.data.mf_pipe_min[p]*(self.variables.T_return[n,t]-self.variables.T_out[p,t])+self.variables.mf_pipe[p,t]*(self.data.T_return_min[n,t]-self.data.T_out_max[p,t])-self.data.mf_pipe_min[p]*(self.data.T_return_min[n,t]-self.data.T_out_max[p,t]),name='T mixing return 2({0},{1})'.format(p,t))         
#
#        self.constraints.T_mixing_supply_3 = {}
#        self.constraints.T_mixing_return_3 = {}
#        
#        for t in time:
#                for n in node: 
#                    for p in self.data.pipe_supply_end[n]:
#                        self.constraints.T_mixing_supply_3[p,t] = m.addConstr(
#                            self.variables.w[p,t],
#                            gb.GRB.LESS_EQUAL,
#                            self.data.mf_pipe_max[p]*(self.variables.T_supply[n,t]-self.variables.T_out[p,t])+self.variables.mf_pipe[p,t]*(self.data.T_supply_min[n,t]-self.data.T_out_max[p,t])-self.data.mf_pipe_max[p]*(self.data.T_supply_min[n,t]-self.data.T_out_max[p,t]),name='T mixing supply 3({0},{1})'.format(p,t))         
#
#                    for p in self.data.pipe_return_end[n]:                        
#                        self.constraints.T_mixing_return_3[p,t] = m.addConstr(
#                            self.variables.w[p,t],
#                            gb.GRB.LESS_EQUAL,
#                            self.data.mf_pipe_max[p]*(self.variables.T_return[n,t]-self.variables.T_out[p,t])+self.variables.mf_pipe[p,t]*(self.data.T_return_min[n,t]-self.data.T_out_max[p,t])-self.data.mf_pipe_max[p]*(self.data.T_return_min[n,t]-self.data.T_out_max[p,t]),name='T mixing return 3({0},{1})'.format(p,t))         
#
#        self.constraints.T_mixing_supply_4 = {}
#        self.constraints.T_mixing_return_4 = {}
#        
#        for t in time:
#                for n in node: 
#                    for p in self.data.pipe_supply_end[n]:
#                        self.constraints.T_mixing_supply_4[p,t] = m.addConstr(
#                            self.variables.w[p,t],
#                            gb.GRB.LESS_EQUAL,
#                            self.data.mf_pipe_min[p]*(self.variables.T_supply[n,t]-self.variables.T_out[p,t])+self.variables.mf_pipe[p,t]*(self.data.T_supply_max[n,t]-self.data.T_out_min[p,t])-self.data.mf_pipe_min[p]*(self.data.T_supply_max[n,t]-self.data.T_out_min[p,t]),name='T mixing supply 4({0},{1})'.format(p,t))         
#
#                    for p in self.data.pipe_return_end[n]:                        
#                        self.constraints.T_mixing_return_4[p,t] = m.addConstr(
#                            self.variables.w[p,t],
#                            gb.GRB.LESS_EQUAL,
#                            self.data.mf_pipe_min[p]*(self.variables.T_return[n,t]-self.variables.T_out[p,t])+self.variables.mf_pipe[p,t]*(self.data.T_return_max[n,t]-self.data.T_out_min[p,t])-self.data.mf_pipe_min[p]*(self.data.T_return_max[n,t]-self.data.T_out_min[p,t]),name='T mixing return 4({0},{1})'.format(p,t))         

                  
        self.constraints.T_in_supply = {}
        self.constraints.T_in_return = {}
        
        for t in time:
                for n in node: 
                    
                    for p in self.data.pipe_supply_start[n]:
                        
                        self.constraints.T_in_supply[p,t] = m.addConstr(
                            self.variables.T_supply[n,t],
                            gb.GRB.EQUAL,
                            self.variables.T_in[p,t],name='T in supply({0},{1})'.format(p,t))         
                    
                    for p in self.data.pipe_return_start[n]:
                        
                        self.constraints.T_in_return[p,t] = m.addConstr(
                            self.variables.T_return[n,t],
                            gb.GRB.EQUAL,
                            self.variables.T_in[p,t],name='T in return({0},{1})'.format(p,t)) 
                            
#        #temp dynamics in pipes
                            
        self.constraints.T_in_init = {}
        
        for p in pipe_supply+pipe_return:
            for t in time_init[p]:
                    
                    self.constraints.T_in_init[p,t] = m.addConstr(
                        self.variables.T_in[p,t],
                        gb.GRB.EQUAL,
                        self.data.T_in_init[p,t],name='T in init({0},{1})'.format(p,t))         

        self.constraints.mf_pipe_init = {}                                    
        for p in pipe_supply+pipe_return:
            for t in time_init[p]:
                    self.constraints.mf_pipe_init[p,t] = m.addConstr(
                        self.variables.mf_pipe[p,t],
                        gb.GRB.EQUAL,
                        self.data.mf_pipe_init[p,t],name='mf pipe init({0},{1})'.format(p,t))  
                        
        self.constraints.time_delay_discretization = {}
        
        for t in time:
                for p in pipe_supply+pipe_return: 
                    
                    self.constraints.time_delay_discretization[p,t] = m.addConstr(
                        self.variables.tau[p,t],
                        gb.GRB.EQUAL,
                        gb.quicksum(n1*(self.variables.u[p,n1,t]-self.variables.u[p,n2,t]) for (n1,n2) in zip(self.data.time_delay_range[p][1:],self.data.time_delay_range[p][:-1])),name='time delay descretization({0},{1})'.format(p,t))         
                            
        
        self.constraints.time_delay_u1 = {}
        self.constraints.time_delay_u2 = {}
        
        for p in pipe_supply+pipe_return: 
            for t in self.data.time_list:
                    for n in self.data.time_delay_range[p]:
                    
                        self.constraints.time_delay_u1[p,n,time[t]] = m.addConstr(
                            -self.data.big_M*(1-self.variables.u[p,n,time[t]]),
                            gb.GRB.LESS_EQUAL,
                            gb.quicksum(self.variables.mf_pipe[p,k] for k in time_extended[p][t+self.data.time_delay_max[p]-n:t+self.data.time_delay_max[p]+1])*self.data.Dt/(np.pi*self.data.radius_pipe[p]*self.data.radius_pipe[p]*self.data.water_density)-self.data.length_pipe[p],name='time delay u1({0},{1})'.format(p,n,time[t]))         
                                                   
                        self.constraints.time_delay_u2[p,n,time[t]] = m.addConstr(
                            self.data.big_M*self.variables.u[p,n,time[t]],
                            gb.GRB.GREATER_EQUAL,
                            self.data.epsilon+gb.quicksum(self.variables.mf_pipe[p,k] for k in time_extended[p][t+self.data.time_delay_max[p]-n:t+self.data.time_delay_max[p]+1])*self.data.Dt/(np.pi*self.data.radius_pipe[p]*self.data.radius_pipe[p]*self.data.water_density)-self.data.length_pipe[p],name='time delay u2({0},{1})'.format(p,n,time[t]))         

        self.constraints.time_delay_v1 = {}
        self.constraints.time_delay_v2 = {}
        self.constraints.time_delay_v3 = {}
        self.constraints.time_delay_v4 = {}
        self.constraints.time_delay_v5 = {}
        self.constraints.time_delay_v6 = {}
        self.constraints.time_delay_v7 = {}
        
        for p in pipe_supply+pipe_return: 
            for t in self.data.time_list:
                    for n in self.data.time_delay_range[p]:
                    
                        
                        #NEW
                        
                        self.constraints.time_delay_v1[p,n,time[t]] = m.addConstr(
                            -self.data.big_M*self.variables.v[p,n,time[t]],
                            gb.GRB.LESS_EQUAL,
                            self.variables.T_in_new[p,n,time[t]],name='time delay v1({0},{1})'.format(p,n,time[t]))         
                        
                        self.constraints.time_delay_v2[p,n,time[t]] = m.addConstr(
                            self.variables.T_in_new[p,n,time[t]],
                            gb.GRB.LESS_EQUAL,
                            self.data.big_M*self.variables.v[p,n,time[t]],name='time delay v2({0},{1})'.format(p,n,time[t]))



                        #NEW

                        self.constraints.time_delay_v3[p,n,time[t]] = m.addConstr(
                            -self.data.big_M*(1-self.variables.v[p,n,time[t]]),
                            gb.GRB.LESS_EQUAL,
                            self.variables.T_in_new[p,n,time[t]]-self.variables.T_in[p,time_extended[p][t+self.data.time_delay_max[p]-n]]*(1-2*n*self.data.water_thermal_loss_coeff/(self.data.water_heat_capacity*self.data.water_density*self.data.radius_pipe[p])),name='time delay v3({0},{1})'.format(p,n,time[t]))         
                        
                        self.constraints.time_delay_v4[p,n,time[t]] = m.addConstr(
                            self.variables.T_in_new[p,n,time[t]]-self.variables.T_in[p,time_extended[p][t+self.data.time_delay_max[p]-n]]*(1-2*n*self.data.water_thermal_loss_coeff/(self.data.water_heat_capacity*self.data.water_density*self.data.radius_pipe[p])),
                            gb.GRB.LESS_EQUAL,
                            self.data.big_M*(1-self.variables.v[p,n,time[t]]),name='time delay v4({0},{1})'.format(p,n,time[t]))

#                        # WITHOUT LOSSES!!!!!!!!!!!!
#                        self.constraints.time_delay_v3[p,n,time[t]] = m.addConstr(
#                            -self.data.big_M*(1-self.variables.v[p,n,time[t]]),
#                            gb.GRB.LESS_EQUAL,
#                            self.variables.T_in_new[p,n,time[t]]-self.variables.T_in[p,time_extended[p][t+self.data.time_delay_max[p]-n]],name='time delay v3({0},{1})'.format(p,n,time[t]))         
#                        
#                        self.constraints.time_delay_v4[p,n,time[t]] = m.addConstr(
#                            self.variables.T_in_new[p,n,time[t]]-self.variables.T_in[p,time_extended[p][t+self.data.time_delay_max[p]-n]],
#                            gb.GRB.LESS_EQUAL,
#                            self.data.big_M*(1-self.variables.v[p,n,time[t]]),name='time delay v4({0},{1})'.format(p,n,time[t]))
#
#
#                        self.constraints.time_delay_v5[p,n,time[t]] = m.addConstr(
#                            -self.data.big_M*(1-self.variables.v[p,n,time[t]]),
#                            gb.GRB.LESS_EQUAL,
#                            n-self.variables.tau[p,time[t]],name='time delay v5({0},{1})'.format(p,n,time[t]))         
#                        
#                        self.constraints.time_delay_v6[p,n,time[t]] = m.addConstr(
#                            n-self.variables.tau[p,time[t]],
#                            gb.GRB.LESS_EQUAL,
#                            self.data.big_M*(1-self.variables.v[p,n,time[t]]),name='time delay v6({0},{1})'.format(p,n,time[t]))


        for p in pipe_supply+pipe_return: 
            for t in self.data.time_list:
                
                    self.constraints.time_delay_v7[p,time[t]] = m.addConstr(
                        gb.quicksum(self.variables.v[p,n,time[t]] for n in self.data.time_delay_range[p]),
                        gb.GRB.EQUAL,
                        1,name='time delay v7({0},{1})'.format(p,time[t]))


#        self.constraints.temp_out = {}
#        
#        for p in pipe_supply+pipe_return: 
#            for t in self.data.time_list:
#                
#                    self.constraints.temp_out[p,time[t]] = m.addConstr(
#                        gb.quicksum(self.variables.T_in[p,time_extended[p][t+self.data.time_delay_max[p]-n]]*self.variables.tau_new[p,n,time[t]] for n in self.data.time_delay_range[p]),
#                        gb.GRB.EQUAL,
#                        self.variables.T_out[p,time[t]])


        self.constraints.temp_out = {}
        
        for p in pipe_supply+pipe_return: 
            for t in time:     
                
                self.constraints.temp_out[p,t] = m.addConstr(
                    gb.quicksum(self.variables.T_in_new[p,k,t] for k in self.data.time_delay_range[p]),
                    gb.GRB.EQUAL,
                    self.variables.T_out[p,t],name='temp out 0({0},{1})'.format(p,t))

                   
        #pressure loss in pipes LP!!!!!!!!! CONVEX
        
#        self.constraints.pressure_loss_supply = {}
#        
#        for t in time:
#                for p in pipe_supply: 
#                    
#                    self.constraints.pressure_loss_supply[p,t] = m.addConstr(
#                        self.variables.pr_supply[self.data.pipe_supply_connexion[p][0],t]-self.variables.pr_supply[self.data.pipe_supply_connexion[p][1],t],
#                        gb.GRB.GREATER_EQUAL,
#                        self.data.pressure_loss_coeff[p]*self.variables.mf_pipe[p,t]*self.variables.mf_pipe[p,t],name='pressure loss supply({0},{1})'.format(p,t))         
#
#        self.constraints.pressure_loss_return = {}
#        
#        for t in time:
#                for p in pipe_return: 
#                    
#                    self.constraints.pressure_loss_return[p,t] = m.addConstr(
#                        self.variables.pr_return[self.data.pipe_return_connexion[p][0],t]-self.variables.pr_return[self.data.pipe_return_connexion[p][1],t],
#                        gb.GRB.GREATER_EQUAL,
#                        self.data.pressure_loss_coeff[p]*self.variables.mf_pipe[p,t]*self.variables.mf_pipe[p,t],name='pressure loss return({0},{1})'.format(p,t))         

        self.constraints.pressure_loss_supply = {}
        
        for t in time:
                for p in pipe_supply: 
                    for l in range(LL):
                    
                        self.constraints.pressure_loss_supply[p,t,l] = m.addConstr(
                            2*np.sqrt(self.data.pressure_loss_coeff[p])*self.variables.mf_pipe[p,t],
                            gb.GRB.LESS_EQUAL,
                            np.sqrt(self.data.pr_supply_slice[l])+(self.variables.pr_supply[self.data.pipe_supply_connexion[p][0],t]-self.variables.pr_supply[self.data.pipe_supply_connexion[p][1],t])/np.sqrt(self.data.pr_supply_slice[l]),name='pressure loss supply({0},{1},{2})'.format(p,t,l))         

        self.constraints.pressure_loss_return = {}

        for t in time:
                for p in pipe_return: 
                    for l in range(LL):
                    
                        self.constraints.pressure_loss_return[p,t,l] = m.addConstr(
                            2*np.sqrt(self.data.pressure_loss_coeff[p])*self.variables.mf_pipe[p,t],
                            gb.GRB.LESS_EQUAL,
                            np.sqrt(self.data.pr_return_slice[l])+(self.variables.pr_return[self.data.pipe_return_connexion[p][0],t]-self.variables.pr_return[self.data.pipe_return_connexion[p][1],t])/np.sqrt(self.data.pr_return_slice[l]),name='pressure loss supply({0},{1},{2})'.format(p,t,l))         
        

        # wind realization

        self.constraints.wind_scenario = {}
        
        for t in time:
                for g in wind: 
                    
                    self.constraints.wind_scenario[g,t] = m.addConstr(
                        self.variables.P[g,t],
                        gb.GRB.LESS_EQUAL,
                        self.data.wind_scenario[g,t]*self.data.elec_maxprod[g],name='wind scenario({0},{1})'.format(n,t))
        
        # elec balance

        self.constraints.elec_balance = {}
        
        for t in time:
                for n in node: 
                    
                    self.constraints.elec_balance[n,t] = m.addConstr(
                        gb.quicksum(self.variables.storage_plus[g,t] - self.variables.storage_moins[g,t] for g in self.data.elec_storage_node[n])+gb.quicksum(self.variables.P[g,t] for g in self.data.elec_station_node[n])+gb.quicksum(self.variables.flow_line[l,t] for l in self.data.line_end[n])-gb.quicksum(self.variables.flow_line[l,t] for l in self.data.line_start[n])-self.data.elec_load[n,t],
                        gb.GRB.EQUAL,
                        0,name='elec balance({0},{1})'.format(n,t))   
        
        # elec transmission constraints   
             
        self.constraints.angle_ref = {}
        
        for t in time:
            self.constraints.angle_ref[t] = m.addConstr(
                        self.variables.node_angle[node[0],t],
                        gb.GRB.EQUAL,
                        0,name='angle ref({0})'.format(t))
      
        self.constraints.elec_flow = {}
        
        for t in time:
                for l in line: 
                    
                    self.constraints.elec_flow[l,t] = m.addConstr(
                        self.variables.flow_line[l,t],
                        gb.GRB.EQUAL,
                        self.data.B[l]*(self.variables.node_angle[self.data.line_connexion[l][0],t]-self.variables.node_angle[self.data.line_connexion[l][1],t]),name='elec flow({0},{1})'.format(l,t))  

#        self.constraints.time_delay_t00 = {}
#        
#        for p in pipe_supply+pipe_return:
#                    
#                self.constraints.time_delay_t00[p] = m.addConstr(
#                    self.variables.tau[p,'t00'],
#                    gb.GRB.LESS_EQUAL,
#                    0,name='time delay t00({0})'.format(p))  

        self.constraints.balance_total = m.addConstr(
                    gb.quicksum(self.variables.Q[h,t] for t in time for h in heat_station+heat_storage),
                    gb.GRB.GREATER_EQUAL,
                    gb.quicksum(self.data.heat_load[h,t] for t in time for h in node),name='balance overday')  
                
                                               
#%% SOLVE                       

dispatch_MILP = integrated_dispatch_MILP()
dispatch_MILP.model.params.OutputFlag = 0
#tstart = datetime.now()
#tstart = tt.time()
dispatch_MILP.optimize()
#tend = tt.time()
#tend = datetime.now()
#exectime=tend-tstart
#dispatch_cost = dispatch.model.ObjVal 

