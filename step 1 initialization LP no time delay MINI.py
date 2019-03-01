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

#%%

pipe=['P{0}'.format(p+1)for p in range(P)]

            
pipe_connexion={'P1':['N1','N2'],'P2':['N2','N3']}
pipe_start = {'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
pipe_end = {'N1':[],'N2':[],'N3':[],'N4':[],'N5':[],'N6':[]}
for n in node:
    for p in pipe:
        if pipe_connexion[p][0]==n:
           pipe_start[n].append(p) 
        if pipe_connexion[p][1]==n:
           pipe_end[n].append(p)
           
#%% DH network parameters

pipe_maxflow = {p:1000 for p in pipe}
