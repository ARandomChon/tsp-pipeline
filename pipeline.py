# -*- coding: utf-8 -*-
"""
Created on Tue Jun  1 08:05:45 2021

@author: vee kalkunte

@author: sean bergen
"""
from teaspoon.parameter_selection.MI_delay import MI_for_delay
from teaspoon.parameter_selection.FNN_n import FNN_n
from teaspoon.parameter_selection.MsPE import MsPE_n,  MsPE_tau
from teaspoon.SP.network import ordinal_partition_graph
from teaspoon.TDA.PHN import PH_network, point_summaries
from teaspoon.SP.network_tools import make_network
from teaspoon.SP.tsa_tools import takens
from ripser import ripser
from persim import plot_diagrams
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import networkx as nx

# specify the data we're looking at:
# this is currently still used for testing
# in future versions this will probably be removed
toImp = "Electric_Production.csv"
varName = 'IPG2211A2N'

"""
tsp_pipeline(ts_data, col_name, takens_flag, permut_flag,
             t_tau, t_d, p_tau, p_d)
    ts_data  -> CSV file/path to CSV containing timeseries data
    col_name -> name of column in CSV where the data is (default = col1)
    takens_f -> should the pipeline produce takens embedding? (default yes)
    permut_f -> should pipeline produce permutation sequence? (default yes)
    t_tau    -> specify a particular takens' embedding tau (default None)
    t_d      -> specify a particular takens' embedding d (default None)
    p_tau    -> specify a particular permutation tau (default None)
    p_d      -> specify a particular permutation d (default None)

this function takes in timeseries data as a CSV and performs steps to generate
ordinal partition network, pairwise distance matrix, and persistant homology
for the data passed in

TODO: add ability to load data in as a dataframe/np array directly instead of
only CSV files

TODO: add checking in the stats for either nan or inf, or some other way to
determine if no homology structure exists from the result of takens or perm

TODO: rewrite the chart section to be seperate function calls for building
either set of charts

when passing a CSV into it, if the column name of the data is not
specified, it will assume that the second column (col1) has the data
that you are tracking

TODO: add something to allow user to specify a particular column as time data
and add functionality to calculate even spaced integers for that

If you do not give column names, the pipeline (as of this time) will assume
that there are no column names at the top of the file

When setting flags for either takens or permutation, they should either be set
to a value of True or False as a best practice

If you want to set your own tau or d (dimension) for a dataset, they
are able to do so through the t_tau, t_d, p_tau, and p_d arguments

TODO: add error catching for invalid tau or d values
"""
def tsp_pipeline(ts_data, col_name=1, takens_f=True, permut_f=True,
                 t_tau=None, t_d=None, p_tau=None,p_d=None):
    # step 1, read in the file
    ts_df = pd.read_csv(ts_data, usecols=[col_name], index_col=False)
    # dataframe to array conversion
#   ts_df = np.array(ts_df)
    ts_df = ts_df.to_numpy()
    ts_df = ts_df.flatten()
    TextSize = 13
    # time for takens/permut
    # check/initialize tau and d
    # and then compute graph for permutation/construct pointcloud for takens
    if(takens_f == True):
        # set tau if a specific one was not set
        if(t_tau == None):
            t_tau = MI_for_delay(ts_df, plotting=False)
        # set d if a specific one was not set
        if(t_d == None):
            perc_FNN, t_d = FNN_n(ts_df, t_tau)

        print("Takens': tau = ",t_tau,", d = ",t_d, sep='')
        # this code was from using a network from takens instead of
        # just a pointcloud, which has now been changed
        # t_adj = ordinal_partition_graph(ts_df, t_d, t_tau)
       
        # this returns the point cloud as numpy array
        t_cloud = takens(ts_df, t_d, t_tau)

        # computing persistence diagrams using ripser
        t_diagram = ripser(t_cloud, maxdim=2)
        
        plt.figure(0)
        plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(4, 2)

        # plot the point cloud if it is 3d or 2d
        # 2d case
        
        if(t_d == 2):
            ax = plt.subplot(2,2,1)
            plt.title('PointCloud', size = TextSize)
            plt.plot(t_cloud[:,0], t_cloud[:,1])
        
        # 3d case
        if(t_d == 3):
            ax = plt.subplot(2,2,1, projection = '3d')
            plt.title('PointCloud', size = TextSize)
            ax.scatter(t_cloud[:,0], t_cloud[:,1], t_cloud[:,2], marker='.')
        
        ax = plt.subplot(2,2,2)
        plot_diagrams(t_diagram["dgms"], show=False)
        

    if(permut_f == True):
        # set tau if a specific one was not set
        if(p_tau == None):
            p_tau = MsPE_tau(ts_df)
        # set d if a specific one was not set
        if(p_d == None):
            p_d = MsPE_n(ts_df, p_tau)

        print("Permutation: tau = ",p_tau,", d = ",p_d, sep='')
        p_adj = ordinal_partition_graph(ts_df, p_d, p_tau)
        p_G, p_pos = make_network(p_adj, remove_deg_zero_nodes=True)
        p_D, p_diagram = PH_network(p_adj)
        p_stats = point_summaries(p_diagram, p_adj)
        print("Stats for Ordinal Partition Network from Permutation Sequence:")
        print(p_stats)

        
        plt.figure(1)
        plt.figure(figsize=(8,8))
        gs = gridspec.GridSpec(4, 2)

        ax = plt.subplot(gs[0:2, 0:2]) #plot time series
        plt.title('Time Series', size = TextSize)
        plt.plot(ts_df, 'k')
        plt.xticks(size = TextSize)
        plt.yticks(size = TextSize)
        plt.xlabel('$t$', size = TextSize)
        plt.ylabel('$x(t)$', size = TextSize)
        plt.xlim(0,len(ts_df))

        ax = plt.subplot(gs[2:4, 0])
        plt.title('Network', size = TextSize)
        nx.draw(p_G, p_pos, with_labels=False,
            font_weight='bold', node_color='blue',
        width=1, font_size = 10, node_size = 30)

        ax = plt.subplot(gs[2:4, 1])
        plt.title('Persistence Diagram', size = TextSize)
        MS = 3
        top = max(p_diagram[1].T[1])
        plt.plot([0,top*1.25],[0,top*1.25],'k--')
        plt.yticks( size = TextSize)
        plt.xticks(size = TextSize)
        plt.xlabel('Birth', size = TextSize)
        plt.ylabel('Death', size = TextSize)
        plt.plot(p_diagram[1].T[0],p_diagram[1].T[1] ,'go', markersize = MS+2)
        plt.xlim(0,top*1.25)
        plt.ylim(0,top*1.25)
    
    # making plots out of data above
    plt.subplots_adjust(hspace= 0.8)
    plt.subplots_adjust(wspace= 0.35)
    plt.show()

tsp_pipeline(toImp, varName)
