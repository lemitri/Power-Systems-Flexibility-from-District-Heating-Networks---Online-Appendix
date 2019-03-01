# -*- coding: utf-8 -*-
"""
Created on Thu Jun  9 08:57:30 2016

@author: lemitri
"""

#%%
 
           
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

os.chdir("C:/Users/lemitri/Google Drive/Combined Heat and Power Flow/python/data")

def produit(list): #takes a list as argument
    p=1
    for x in list:
        p=p*x
    return(p)
 
#%% indexes
    
T=24
N=6
G=2 #elec generator
W=1 #wind
H=1 #chp
HO=1 #heat only
HS=0 #storage
ES=0
HP=1 #heat pump
#HES=2
P=2
L=7
LL=500

#%%

time = ['t{0:02d}'.format(t+0) for t in range(T)]
time_day =  ['t{0:02d}'.format(t) for t in range(24)] # optimization periods indexes (24 hours)
time_list=np.arange(T)
node=['N{0}'.format(n+1)for n in range(N)]
line=['L{0}'.format(l+1)for l in range(L)]
pipe_supply=['PS{0}'.format(p+1)for p in range(P)]
pipe_return=['PR{0}'.format(p+1)for p in range(P)]
gen=['G{0}'.format(g+1) for g in range(G)]
heat_storage=['HS{0}'.format(s+1) for s in range(HS)]
elec_storage=['ES{0}'.format(s+1) for s in range(ES)]
heat_pump = ['HP{0}'.format(h+1) for h in range(HP)]
wind=['W{0}'.format(w+1) for w in range(W)]
heat_only = ['HO{0}'.format(ho+1) for ho in range(HO)]
#heat_exchanger_station = ['HES{0}'.format(h+1) for h in range(HES)]
CHP_sorted = {'ex':['CHP1'],'bp':[]} # CHPs indexes sorted by type: extraction or backpressure
CHP = list(it.chain.from_iterable(CHP_sorted.values()))

# heat market data
heat_station=CHP+heat_only+heat_pump
elec_station=CHP+gen+wind+heat_pump
producers=CHP+heat_only+heat_storage+heat_pump+gen+wind+elec_storage

producers_node={'N1':['HP1','W1'],'N2':['CHP1','HO1','G2'],'N3':[],'N4':[],'N5':[],'N6':['G1']}
#heat_exchanger_station_node = {'N1':[],'N2':[],'N3':['HES1'],'N4':[],'N5':[],'N6':['HES2']}
elec_station_node={'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
heat_station_node={'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
CHP_node={'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
CHP_sorted_node={'N1':{'ex':[],'bp':[]},'N2':{'ex':[],'bp':[]},'N3':{'ex':[],'bp':[]},'N4':{'ex':[],'bp':[]},'N5':{'ex':[],'bp':[]},'N6':{'ex':[],'bp':[]}}
heat_pump_node ={'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
heat_only_node ={'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
heat_storage_node={'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
gen_node ={'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
wind_node={'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
elec_storage_node ={'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}


for n in node:
    for h in CHP:
        if h in producers_node[n]:
            CHP_node[n].append(h)
            heat_station_node[n].append(h)
            elec_station_node[n].append(h)
            if h in CHP_sorted['ex']:
                CHP_sorted_node[n]['ex'].append(h)
            if h in CHP_sorted['bp']:
                CHP_sorted_node[n]['bp'].append(h)
    for h in heat_only:
        if h in producers_node[n]:
            heat_only_node[n].append(h)
            heat_station_node[n].append(h)
    for h in heat_storage:
        if h in producers_node[n]:
            heat_storage_node[n].append(h)
            #heat_station_node[n].append(h)
    for h in heat_pump:
        if h in producers_node[n]:
            heat_pump_node[n].append(h)
            heat_station_node[n].append(h)
            elec_station_node[n].append(h)
    for h in gen:
        if h in producers_node[n]:
            gen_node[n].append(h) 
            elec_station_node[n].append(h)
    for h in wind:
        if h in producers_node[n]:
            wind_node[n].append(h) 
            elec_station_node[n].append(h)
    for h in elec_storage:
        if h in producers_node[n]:
            elec_storage_node[n].append(h) 
            #elec_station_node[n].append(h)
            
pipe_supply_connexion={'PS1':['N1','N2'],'PS2':['N2','N3']}
pipe_supply_start = {'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
pipe_supply_end = {'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
for n in node:
    for p in pipe_supply:
        if pipe_supply_connexion[p][0]==n:
           pipe_supply_start[n].append(p) 
        if pipe_supply_connexion[p][1]==n:
           pipe_supply_end[n].append(p)

pipe_return_connexion={pipe_return[p]:[pipe_supply_connexion[pipe_supply[p]][1],pipe_supply_connexion[pipe_supply[p]][0]] for p in range(P)}

pipe_return_start = {'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
pipe_return_end = {'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}

for n in node:
    for p in pipe_return:
        if pipe_return_connexion[p][0]==n:
           pipe_return_start[n].append(p) 
        if pipe_return_connexion[p][1]==n:
           pipe_return_end[n].append(p)
           
line_connexion={'L1':['N1','N2'],'L2':['N2','N3'],'L3':['N3','N6'],'L4':['N6','N5'],'L5':['N5','N4'],'L6':['N4','N1'],'L7':['N3','N5']}
line_start = {'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
line_end = {'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
for n in node:
    for l in line:
        if line_connexion[l][0]==n:
           line_start[n].append(l) 
        if line_connexion[l][1]==n:
           line_end[n].append(l)   
           
           
#%% loads
           
#heat=pd.read_csv("heat_demand_Tingbjerg_2015.csv",sep=";",decimal=",")
#heat_load = {'N1':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N2':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N3':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':}}
heat_load = {}
for t in time:
    heat_load['N1',t]=0
    heat_load['N2',t]=0
    heat_load['N4',t]=0
    heat_load['N5',t]=0
    heat_load['N6',t]=0

heat_load_csv=pd.read_csv("heat_demand_Tingbjerg_2015.csv",sep=";",decimal=",")

heat_load_0 = {time_day[t]  : heat_load_csv.ix[t+1*24,1] for t in range(24)}

for t in time:
    heat_load['N3',t]=heat_load_0[t]*100/max(heat_load_0[t] for t in time)

#heat_load['N3', 't07']= 93.354037267080734
#heat_load['N3', 't08']= 92.11180124223602
#heat_load['N3', 't09']= 91.987577639751549
#heat_load['N3', 't10']= 61.440993788819867
#heat_load['N3', 't11']= 51.459627329192543 
#heat_load['N3', 't12']= 100

heat_load[ 'N3', 't23'] = 82.422360248447205
heat_load[ 'N3', 't00'] = 78.012422360248436
heat_load[ 'N3', 't01'] = 72.298136645962728
heat_load[ 'N3', 't02'] = 72.298136645962728
heat_load[ 'N3', 't03'] = 72.385093167701854
heat_load[ 'N3', 't04'] = 82.608695652173907+10
heat_load[ 'N3', 't05' ] =  83.354037267080734+10 
heat_load[ 'N3', 't06'] = 85.590062111801231+10
heat_load['N3', 't07']=  86.086956521739125
heat_load['N3', 't08']= 92.11180124223602
heat_load['N3', 't09']= 71.987577639751549
heat_load['N3', 't10']= 69.440993788819867
heat_load['N3', 't11']= 66.459627329192543
heat_load['N3', 't12']= 64.720496894409933
heat_load['N3', 't13']= 63.354037267080734 
heat_load['N3', 't14']= 62.11180124223602 
heat_load['N3', 't15']= 61.987577639751549 
heat_load['N3', 't16']= 64.968944099378874
heat_load['N3', 't17']= 64.472049689440993
heat_load['N3', 't18']= 66.583850931677006
heat_load['N3', 't19']= 110.987577639751549
heat_load['N3', 't20']= 97.440993788819867
heat_load['N3', 't21']= 95.459627329192543 
heat_load['N3', 't22']= 85.590062111801231


#heat_load[ 'N3', 't23'] = 82.422360248447205
#heat_load[ 'N3', 't00'] = 78.012422360248436
#heat_load[ 'N3', 't01'] = 72.298136645962728
#heat_load[ 'N3', 't02'] = 72.298136645962728
#heat_load[ 'N3', 't03'] = 72.385093167701854
#heat_load[ 'N3', 't04'] = 82.608695652173907+10
#heat_load[ 'N3', 't05' ] =  83.354037267080734+10 
#heat_load[ 'N3', 't06'] = 85.590062111801231+10
#heat_load['N3', 't07']=  86.086956521739125
#heat_load['N3', 't08']= 92.11180124223602
#heat_load['N3', 't09']= 71.987577639751549
#heat_load['N3', 't10']= 69.440993788819867
#heat_load['N3', 't11']= 66.459627329192543
#heat_load['N3', 't12']= 64.720496894409933
#heat_load['N3', 't13']= 63.354037267080734 
#heat_load['N3', 't14']= 62.11180124223602 
#heat_load['N3', 't15']= 61.987577639751549 
#heat_load['N3', 't16']= 64.968944099378874
#heat_load['N3', 't17']= 64.472049689440993
#heat_load['N3', 't18']= 66.583850931677006
#heat_load['N3', 't19']= 110.987577639751549
#heat_load['N3', 't20']= 97.440993788819867
#heat_load['N3', 't21']= 95.459627329192543 
#heat_load['N3', 't22']= 85.590062111801231

#heat_load['N3','t00']=0
#heat_load['N3','t01']=0
#heat_load['N3','t02']=0
#heat_load['N3','t03']=0
#heat_load['N3','t04']=0
#heat_load['N3','t05']=0
#heat_load['N3','t06']=0
#heat_load['N3','t07']=0
#heat_load['N3','t08']=0
#heat_load['N3','t09']=0
#heat_load['N3','t10']=0
#heat_load['N3','t11']=0
#heat_load['N3','t12']=0
#heat_load['N3','t13']=0
#heat_load['N3','t14']=0
#heat_load['N3','t15']=0
#heat_load['N3','t16']=0
#heat_load['N3','t17']=0
#heat_load['N3','t18']=0
#heat_load['N3','t19']=0
#heat_load['N3','t20']=0
#heat_load['N3','t21']=0
#heat_load['N3','t22']=0
#heat_load['N3','t23']=0
#
#heat_load['N6','t00']=0
#heat_load['N6','t01']=0
#heat_load['N6','t02']=0
#heat_load['N6','t03']=0
#heat_load['N6','t04']=0
#heat_load['N6','t05']=0
#heat_load['N6','t06']=0
#heat_load['N6','t07']=0
#heat_load['N6','t08']=0
#heat_load['N6','t09']=0
#heat_load['N6','t10']=0
#heat_load['N6','t11']=0
#heat_load['N6','t12']=0
#heat_load['N6','t13']=0
#heat_load['N6','t14']=0
#heat_load['N6','t15']=0
#heat_load['N6','t16']=0
#heat_load['N6','t17']=0
#heat_load['N6','t18']=0
#heat_load['N6','t19']=0
#heat_load['N6','t20']=0
#heat_load['N6','t21']=0
#heat_load['N6','t22']=0
#heat_load['N6','t23']=0

#elec_load_mean = {'t00':780,'t01':750,'t02':730,'t03':770,'t04':800,'t05':850,'t06':1000,'t07':1200,'t08':1400,'t09':1300,'t10':1280,'t11':1250,'t12':1230,'t13':1100,'t14':1050,'t15':1000,'t16':980,'t17':950,'t18':1010,'t19':1100,'t20':980,'t21':930,'t22':850,'t23':830}
#elec_load = {'N1':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N2':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N3':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':}}

elec_load_IEEE = {'t00':820,'t01':820,'t02':815,'t03':815,'t04':810,'t05':850,'t06':1000,'t07':1150+100*1400/350 ,'t08':1250+100*1400/350 ,'t09':1250+100*1400/350 ,'t10':1100+100*1400/350,'t11':1000+100*1400/350,'t12':1000+50*1400/350,'t13':955+50*1400/350,'t14':950+50*1400/350,'t15':950+36*1400/350,'t16':900+25*1400/350,'t17':950,'t18':1010,'t19':1100,'t20':1125,'t21':1025,'t22':950,'t23':850}


elec_load = {}
for t in time:
    elec_load['N1',t]=0
    elec_load['N2',t]=0
    elec_load['N3',t]= elec_load_IEEE[t]*0.20*350/1400
    elec_load['N4',t]= elec_load_IEEE[t]*0.40*350/1400
    elec_load['N5',t]= elec_load_IEEE[t]*0.40*350/1400
    elec_load['N6',t]=0


   
#elec_load['N2','t00']=0
#elec_load['N2','t01']=0
#elec_load['N2','t02']=0
#elec_load['N2','t03']=0
#elec_load['N2','t04']=0
#elec_load['N2','t05']=0
#elec_load['N2','t06']=0
#elec_load['N2','t07']=0
#elec_load['N2','t08']=0
#elec_load['N2','t09']=0
#elec_load['N2','t10']=0
#elec_load['N2','t11']=0
#elec_load['N2','t12']=0
#elec_load['N2','t13']=0
#elec_load['N2','t14']=0
#elec_load['N2','t15']=0
#elec_load['N2','t16']=0
#elec_load['N2','t17']=0
#elec_load['N2','t18']=0
#elec_load['N2','t19']=0
#elec_load['N2','t20']=0
#elec_load['N2','t21']=0
#elec_load['N2','t22']=0
#elec_load['N2','t23']=0
#
#elec_load['N3','t00']=0
#elec_load['N3','t01']=0
#elec_load['N3','t02']=0
#elec_load['N3','t03']=0
#elec_load['N3','t04']=0
#elec_load['N3','t05']=0
#elec_load['N3','t06']=0
#elec_load['N3','t07']=0
#elec_load['N3','t08']=0
#elec_load['N3','t09']=0
#elec_load['N3','t10']=0
#elec_load['N3','t11']=0
#elec_load['N3','t12']=0
#elec_load['N3','t13']=0
#elec_load['N3','t14']=0
#elec_load['N3','t15']=0
#elec_load['N3','t16']=0
#elec_load['N3','t17']=0
#elec_load['N3','t18']=0
#elec_load['N3','t19']=0
#elec_load['N3','t20']=0
#elec_load['N3','t21']=0
#elec_load['N3','t22']=0
#elec_load['N3','t23']=0
#
#elec_load['N4','t00']=0
#elec_load['N4','t01']=0
#elec_load['N4','t02']=0
#elec_load['N4','t03']=0
#elec_load['N4','t04']=0
#elec_load['N4','t05']=0
#elec_load['N4','t06']=0
#elec_load['N4','t07']=0
#elec_load['N4','t08']=0
#elec_load['N4','t09']=0
#elec_load['N4','t10']=0
#elec_load['N4','t11']=0
#elec_load['N4','t12']=0
#elec_load['N4','t13']=0
#elec_load['N4','t14']=0
#elec_load['N4','t15']=0
#elec_load['N4','t16']=0
#elec_load['N4','t17']=0
#elec_load['N4','t18']=0
#elec_load['N4','t19']=0
#elec_load['N4','t20']=0
#elec_load['N4','t21']=0
#elec_load['N4','t22']=0
#elec_load['N4','t23']=0

#%% DH network parameters

Dt=60*60 #sec/hour
#pipelines parameters

# length of pipes in in meter
length_pipe = {p:500 for p in pipe_supply+pipe_return} # in m

fanning_coeff={p:0.05/4 for p in pipe_supply+pipe_return} # NO UNIT (moody chart)
radius_pipe = {p:0.8 for p in pipe_supply+pipe_return} # in m
water_density = 1000 # in kg/m3
water_thermal_loss_coeff = 20 # in W/m2.K
water_heat_capacity = 4.18*0.28 # in Wh/kg.K
pressure_loss_coeff = {p:2*16/(np.pi**2)*fanning_coeff[p]*length_pipe[p]/(((radius_pipe[p]*2)**5)*water_density) for p in pipe_supply+pipe_return}

#pressure bounds in Pa
pr_return_max = {n:5000 for n in node}
pr_return_min = {n:50 for n in node}
pr_supply_max = {n:5000 for n in node}
pr_supply_min = {n:50 for n in node}
pr_diff_min = {n:1000 for n in node}


#OUTER APPROX SLICES

pr_return_slice=[pr_return_min[n]+l*(pr_return_max[n]-pr_return_min[n])/LL for l in range(LL)]   
pr_supply_slice=[pr_supply_min[n]+l*(pr_supply_max[n]-pr_supply_min[n])/LL for l in range(LL)]   


#mass flow bounds
    
#mf_pipe_max = {p:2*0.062*water_density*np.pi*(radius_pipe[p]**2) for p in pipe_supply+pipe_return}
mf_pipe_min = {p:50 for p in pipe_supply+pipe_return}

mf_pipe_max = {}
mf_pipe_max['PS1']=300 #300
mf_pipe_max['PS2']=300 #300


mf_pipe_max['PR1']=mf_pipe_max['PS1']
mf_pipe_max['PR2']=mf_pipe_max['PS2']

mf_HS_max = {}
for h in heat_station+heat_storage:
    mf_HS_max[h] = 0 #500
mf_HS_max['CHP1'] = 301 #500
mf_HS_max['HP1'] = 300 #500
mf_HS_min = {} #carefull for HEAT STORAGE!!!!
for h in heat_station:
    mf_HS_min[h] = 0
for h in heat_storage:
    mf_HS_min[h] = 0
mf_HES_max  = {} 
mf_HES_max['N1']=0
mf_HES_max['N2']=0
mf_HES_max['N3']=300 #300
mf_HES_max['N4']=0
mf_HES_max['N5']=0
mf_HES_max['N6']=0
mf_HES_min  = {}
mf_HES_min['N1']=0
mf_HES_min['N2']=0
mf_HES_min['N3']=150
mf_HES_min['N4']=0
mf_HES_min['N5']=0
mf_HES_min['N6']=0


# time delay
time_delay_max = {p:int(water_density*length_pipe[p]*np.pi*(radius_pipe[p]**2)/(Dt*mf_pipe_min[p]))+1 for p in pipe_supply+pipe_return} #in hour
time_delay_range = {p:[x for x in range(time_delay_max[p]+1)] for p in pipe_supply+pipe_return}

time_init = {p:['t00-{0}'.format(time_delay_max[p]-k) for k in time_delay_range[p][:-1]] for p in pipe_supply+pipe_return}
time_extended = {p:time_init[p]+time for p in pipe_supply+pipe_return}

tau_new_min = {p:min(0,1-2*time_delay_max[p]*water_thermal_loss_coeff/(water_heat_capacity*water_density*radius_pipe[p])) for p in pipe_supply+pipe_return}
tau_new_max = {p:1 for p in pipe_supply+pipe_return}

# time delay
time_delay_max = {p:int(water_density*length_pipe[p]*np.pi*(radius_pipe[p]**2)/(Dt*mf_pipe_min[p]))+1 for p in pipe_supply+pipe_return} #in hour
time_delay_range = {p:[x for x in range(time_delay_max[p]+1)] for p in pipe_supply+pipe_return}

time_init = {p:['t00-{0}'.format(time_delay_max[p]-k) for k in time_delay_range[p][:-1]] for p in pipe_supply+pipe_return}
time_extended = {p:time_init[p]+time for p in pipe_supply+pipe_return}

tau_new_min = {p:min(0,1-2*time_delay_max[p]*water_thermal_loss_coeff/(water_heat_capacity*water_density*radius_pipe[p])) for p in pipe_supply+pipe_return}
tau_new_max = {p:1 for p in pipe_supply+pipe_return}


#Initial data: from XX last hours of the day before

mf_pipe_init = {}
for p in pipe_supply+pipe_return:
    for t in time_init[p]:
        mf_pipe_init[p,t]=mf_pipe_min[p]
    
T_in_init = {}
for p in pipe_supply:
    for t in time_init[p]:
        T_in_init[p,t] = 90 + 273.15
for p in pipe_return:
    for t in time_init[p]:
        T_in_init[p,t] = 60 + 273.15

#temperature bounds
T_return_max = {}
T_return_min = {}
T_supply_max = {}
T_supply_min = {} 

for t in time_extended['PS1']:
    for n in node:
        T_supply_max[n,t]=120  + 273.15
        T_supply_min[n,t]=90  + 273.15
        T_return_max[n,t]=60  + 273.15
        T_return_min[n,t]=30  + 273.15
    
T_in_max = {}
T_in_min = {}
T_out_max = {}
T_out_min = {}

for p in pipe_supply:
    for t in time_extended[p]: 
        T_in_max[p,t]=T_supply_max[pipe_supply_connexion[p][0],t]
        T_in_min[p,t]=T_supply_min[pipe_supply_connexion[p][0],t]
        T_out_max[p,t]=T_supply_max[pipe_supply_connexion[p][1],t]
        T_out_min[p,t]=T_supply_min[pipe_supply_connexion[p][1],t]
for p in pipe_return:
    for t in time_extended[p]: 
        T_in_max[p,t]=T_return_max[pipe_return_connexion[p][0],t]       
        T_in_min[p,t]=T_return_min[pipe_return_connexion[p][0],t] 
        T_out_max[p,t]=T_return_max[pipe_return_connexion[p][1],t]       
        T_out_min[p,t]=T_return_min[pipe_return_connexion[p][1],t] 

#%% cost

alpha = {}
for t in time:
    alpha['CHP1',t]=12.5
    alpha['HO1',t]=80
    alpha['G1',t]=11
    alpha['G2',t]=33
    alpha['W1',t]=0.0001
    
#%% technical characteristics
          
heat_maxprod = {'CHP1': 100,'HP1':150,'HO1':0} #HP 150
rho_elec = {'CHP1': 2.4} # efficiency of the CHP for electricity production
rho_heat = {'CHP1': 0.25,'HO1':1} # efficiency of the CHP for heat production
r_min = {'CHP1' : 0.6} # elec/heat ratio (flexible in the case of extraction units) 
CHP_maxprod = {'CHP1': 250}

COP={'HP1':2.5}

storage_loss={h:10 for h in heat_storage+elec_storage}
storage_init={h:1000 for h in heat_storage+elec_storage}
storage_rho_plus={h:1.1 for h in heat_storage+elec_storage} # >=1
storage_rho_moins={h:0.9 for h in heat_storage+elec_storage} # <=1
storage_maxcapacity={h:2000 for h in heat_storage+elec_storage}
storage_maxprod = {h:500 for h in heat_storage+elec_storage}

elec_maxprod = {'CHP1':1000,'G1':180,'G2':100,'W1':500} # known

pp = pd.Panel({'W' + str(i+1):pd.read_csv('scen_zone{0}.out'.format(i+1), index_col=0) for i in range(W)})

#%%

wind_scenario = {}

#for w in wind:
#    for t in range(T):
#        wind_scenario[w,time[t]] = pp[w,t+1,'V2'] 
                
wind_scenario['W1','t00'] = 0.4994795107848
wind_scenario['W1','t01'] = 0.494795107848
wind_scenario['W1','t02'] = 0.494795107848
wind_scenario['W1','t03'] = 0.505243011484
wind_scenario['W1','t04'] = 0.53537368424
wind_scenario['W1','t05'] = 0.555562455471
wind_scenario['W1','t06'] = 0.628348636916
wind_scenario['W1','t07'] = 0.6461954549
wind_scenario['W1','t08'] = 0.622400860956
wind_scenario['W1','t09'] = 0.580111023006
wind_scenario['W1','t10'] = 0.714935503018
wind_scenario['W1','t11'] = 0.824880140759
wind_scenario['W1','t12'] = 0.416551027874
wind_scenario['W1','t13'] = 0.418463919582
wind_scenario['W1','t14'] = 0.39525842857
wind_scenario['W1','t15'] = 0.523097379857
wind_scenario['W1','t16'] = 0.476699300008
wind_scenario['W1','t17'] = 0.626077589123
wind_scenario['W1','t18'] = 0.684294396661
wind_scenario['W1','t19'] = 0.0598119722706 
wind_scenario['W1','t20'] = 0.0446453658917 
wind_scenario['W1','t21'] = 0.485237701755
wind_scenario['W1','t22'] = 0.49466503395
wind_scenario['W1','t23'] = 0.4993958131342

#supply_maxtwmp = {'N1':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N2':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N3':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':}}
#supply_mintemp = {'N1':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N2':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N3':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':}}
#return_maxtwmp = {'N1':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N2':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N3':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':}}
#return_mintemp = {'N1':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N2':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':},'N3':{'t00':,'t01':,'t02':,'t03':,'t04':,'t05':,'t06':,'t07':,'t08':,'t09':,'t10':,'t11':,'t12':,'t13':,'t14':,'t15':,'t16':,'t17':,'t18':,'t19':,'t20':,'t21':,'t22':,'t23':}}

#susceptance in siemens
B={}
line_maxflow = {}

B['L1'] = 1/0.170
B['L2'] = 1/0.037
B['L3'] = 1/0.258
B['L4'] = 1/0.197
B['L5'] = 1/0.037
B['L6'] = 1/0.140
B['L7'] = 1/0.018

line_maxflow['L1'] = 400
line_maxflow['L2'] = 200
line_maxflow['L3'] = 200
line_maxflow['L4'] = 200
line_maxflow['L5'] = 200
line_maxflow['L6'] = 200
line_maxflow['L7'] = 200


#%%
print('NET elec load')
for t in time:
    print(t,sum(elec_load[n,t] for n in node)-wind_scenario['W1',t]*elec_maxprod['W1'])
print('NET elec load')
for t in time:    
    print(t,sum(heat_load[n,t] for n in node))