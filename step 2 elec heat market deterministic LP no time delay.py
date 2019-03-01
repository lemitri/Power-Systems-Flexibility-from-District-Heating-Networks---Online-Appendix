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
    
#%% building the STOCHSTICA MPEC optimization problem (GUROBI) = heat market clearing

class expando(object):
    '''
    
    
        A small class which can have attributes set
    '''
    pass

class integrated_dispatch_LP:
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
        self.data.pipe=pipe
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
        self.data.pipe_connexion= pipe_connexion
        self.data.pipe_start = pipe_start
        self.data.pipe_end = pipe_end
        self.data.line_connexion= line_connexion
        self.data.line_start = line_start
        self.data.line_end = line_end
        
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
        
        # DHN parameters
        self.data.pipe_maxflow = pipe_maxflow 
        
        # Elec station parameters
        self.data.elec_maxprod = elec_maxprod
        self.data.wind_scenario = wind_scenario
        
        # Cost parameters
        self.data.alpha = alpha    
        

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
        node=self.data.node
        line=self.data.line
        pipe=self.data.pipe
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

        self.variables.flow_pipe = {}
        for t in time:
            for p in pipe:
                self.variables.flow_pipe[p,t] = m.addVar(lb=0,ub=self.data.pipe_maxflow[p],name='flow pipe({0},{1})'.format(p,t))                    
 
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
                
        # electricity transmission system variables 

        self.variables.node_angle = {}
        for t in time:
            for n in node:
                    self.variables.node_angle[n,t] = m.addVar(lb=-gb.GRB.INFINITY,name='node angle({0},{1})'.format(n,t))

        self.variables.flow_line = {}
        for t in time:
            for l in line:
                self.variables.flow_line[l,t] = m.addVar(lb=-self.data.line_maxflow[l],ub=self.data.line_maxflow[l],name='flow line({0},{1})'.format(l,t))                    
 
        m.update()
    
    def _build_objective(self): # building the objective function for the heat maret clearing

        #indexes shortcuts 
        time = self.data.time
        node=self.data.node
        line=self.data.line
        pipe=self.data.pipe
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
              
        m.setObjective(gb.quicksum(self.data.alpha[g,t]*self.variables.P[g,t] for t in time for g in gen+wind)+gb.quicksum(self.data.alpha[g,t]*self.variables.Q[g,t] for t in time for g in heat_only)+gb.quicksum(self.data.alpha[g,t]*(self.data.rho_elec[g]*self.variables.P[g,t]+self.data.rho_heat[g]*self.variables.Q[g,t]) for t in time for g in CHP),   
            gb.GRB.MINIMIZE)
            
        
    def _build_constraints(self):

        #indexes shortcuts 
        time = self.data.time
        node=self.data.node
        line=self.data.line
        pipe=self.data.pipe
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
        
        # heat balance!!  

        self.constraints.heat_balance = {}
        
        for t in time:
            for n in node:
                    
                    self.constraints.heat_balance[n,t] = m.addConstr(
                        gb.quicksum(self.variables.storage_plus[i,t]-self.variables.storage_moins[i,t] for i in self.data.heat_storage_node[n]) + gb.quicksum(self.variables.Q[i,t] for i in self.data.heat_station_node[n])+gb.quicksum(self.variables.flow_pipe[p,t] for p in self.data.pipe_end[n])-gb.quicksum(self.variables.flow_pipe[p,t] for p in self.data.pipe_start[n]),
                        gb.GRB.EQUAL,
                        self.data.heat_load[n,t],name='heat balance({0},{1})'.format(n,t))    

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
                                               
#%% SOLVE                       

dispatch_LP = integrated_dispatch_LP()
dispatch_LP.model.params.OutputFlag = 1
#tstart = datetime.now()
#tstart = tt.time()
dispatch_LP.optimize()
#tend = tt.time()
#tend = datetime.now()
#exectime=tend-tstart
#dispatch_cost = dispatch.model.ObjVal 

