# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:44:51 2021

@author: saipr
"""

import numpy as np
import math
from ripser import ripser#, plot_diagrams
from persim import plot_diagrams
from scipy.interpolate import interp1d

#converted directly from: https://github.com/joperea/sw1pers/blob/master/sw1pers_v1_src/SW1PerS_v1.m

#a bit of a bare minimum conversion - missing a few of the options
def score(signal, num_cycles, num_points, feature_type = False, allow_trending = False, use_meanshift = False, meanshift_epsilon = 0, use_expaverage = False, expaverage_alpha = 0, use_movingaverage = False, movingaverage_window = 0, use_smoothspline = False, smoothspline_sigma = 0):
    nS = len(signal)
    #signal = np.reshape(signal, 1) #og mathlab code: signal = reshape(signal,1,[]);
    
    #define parameters (Taken from code directly)
    N = 7
    p = 11
    M = 2*N
    tau = (2*math.pi) / ((M + 1)*num_cycles)
    

    #signal pre-processings
    #moving average
    if(use_expaverage):
        sigLowPass = signal
        for i in range(2, nS):
            sigLowPass[i] = sigLowPass[i-1] + (expaverage_alpha)*(signal(i) - sigLowPass(i-1))
        signal = sigLowPass
    
    #smooth low pass demolishing via moving average - not implemented, not sure how to
	#if use_movingaverage:
		#signal = smooth(signal',movingaverage_window,'moving')        

    #detrending - skipped over, part of allow_trending boolean option
        
    #create Sliding Window Point Cloud Data
    t = 2*math.pi*np.linspace(0,1,nS)

    T = (2*math.pi - M*tau)*np.linspace(0,1,num_points)
    this = range(0, M)
    tt = tau*this*np.ones(1,num_points) + np.ones(M+1,1)*T
    
    use_smoothspline = False
    #Smooth splining
    #if use_smoothspline:
        #ss.weights = np.ones(1,nS)
        #tolerance = nS*(smoothspline_sigma*peak2peak(signal))**2
        #[sp, vals] =  spaps(t,signal,tolerance, ss.weights)
        #cloud_raw = fnval(sp,tt)
        #signal = vals
    #else:
    cloud_raw = interp1d(t, signal , tt)
    diagram = ripser(cloud_raw, maxdim=2)
    dgms = diagram['dgms']
    plot_diagrams(dgms, show = True)
        
    #now center the cloud:
    #if allow_trending:
        #cloud_centered = cloud_raw - np.ones(M+1,1)*np.mean(cloud_raw)
    #else:
        #cloud_centered = cloud_raw - np.mean(signal)
        
    #and then normalize the cloud
    #SW_cloud = cloud_centered/(np.ones(M+1,1)*math.sqrt(sum(cloud_centered**2))) 
    
    #throw the point cloud into Ripser
    #diagram = ripser(SW_cloud, maxdim=2)
    #dgms = diagram['dgms']
    #plot_diagrams(dgms, show = True)
    
    return None

import pandas as pd
data = 'hades.csv'
ts_df = pd.read_csv(data)
players = ts_df['Players'].tolist()
print(score(players, 50, len(players)))

