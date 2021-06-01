# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 08:05:45 2021

@author: saipr

def viewLag(x, maxLag = 10):
    lagDict = {}
    lags = np.array([i for i in range(1, maxLag)]) #all the lags we're testing for
    for l in lags:
        y = 
"""
import teaspoon as tsp
from teaspoon.parameter_selection.MI_delay import MI_for_delay
from teaspoon.parameter_selection.FNN_n import FNN_n
from teaspoon.parameter_selection.MsPE import MsPE_n,  MsPE_tau
from teaspoon.SP.network import ordinal_partition_graph
from teaspoon.TDA.PHN import PH_network
from teaspoon.SP.network_tools import make_network
from teaspoon.parameter_selection.MsPE import MsPE_tau
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

#specify the data we're looking at:
toImp = "Electric_Production.csv"
varName = 'IPG2211A2N'
timeSeries = pd.read_csv(toImp, index_col=False)
x = np.array(timeSeries[varName])

#----------------------------Takens/Permutation-------------------------------#
#import and find its tau via min. mutual information
tau = MI_for_delay(x, plotting = False) #lag = 4 for this one
#find embedding dimension by false nearest neighbors
perc_FNN, n= FNN_n(x, tau) #n is embedding dimension

#tau and D are chosen via permutation entropy
tau2 = int(MsPE_tau(x, plotting = False))
n2 = MsPE_n(x, tau2)
#print(tau2, n2) #picks a tau of 2, n of 4

#----------------------------Ordinal Partition Network------------------------#
#now to build an adjacency network
adj = ordinal_partition_graph(x, n, tau) #using takens'
#get networkx representations for plotting
G, pos = make_network(adj, remove_deg_zero_nodes = True)
#create distance matrix and persistance diagram
D, diagram = PH_network(adj)

#persistance metrics -- 
stats = point_summaries(diagram, adj)
print(stats) #divide by 0 error

#----------------------------Plot the networks / graphs-----------------------#
TextSize = 13
plt.figure(2)
plt.figure(figsize=(8,8))
gs = gridspec.GridSpec(4, 2)

ax = plt.subplot(gs[0:2, 0:2]) #plot time series
plt.title('Time Series', size = TextSize)
plt.plot(x, 'k')
plt.xticks(size = TextSize)
plt.yticks(size = TextSize)
plt.xlabel('$t$', size = TextSize)
plt.ylabel('$x(t)$', size = TextSize)
plt.xlim(0,len(x))

ax = plt.subplot(gs[2:4, 0])
plt.title('Network', size = TextSize)
nx.draw(G, pos, with_labels=False, font_weight='bold', node_color='blue',
        width=1, font_size = 10, node_size = 30)

ax = plt.subplot(gs[2:4, 1])
plt.title('Persistence Diagram', size = TextSize)
MS = 3
top = max(diagram[1].T[1])
plt.plot([0,top*1.25],[0,top*1.25],'k--')
plt.yticks( size = TextSize)
plt.xticks(size = TextSize)
plt.xlabel('Birth', size = TextSize)
plt.ylabel('Death', size = TextSize)
plt.plot(diagram[1].T[0],diagram[1].T[1] ,'go', markersize = MS+2)
plt.xlim(0,top*1.25)
plt.ylim(0,top*1.25)

plt.subplots_adjust(hspace= 0.8)
plt.subplots_adjust(wspace= 0.35)
plt.show()

