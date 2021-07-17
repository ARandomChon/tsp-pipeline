# -*- coding: utf-8 -*-
"""
Created on Thu Jul  1 09:44:51 2021

@author: saipr
"""

import numpy as np
import math
from persim import plot_diagrams
from scipy.interpolate import CubicSpline
from scipy.spatial.distance import pdist, squareform, cosine
from ripser import ripser
import matplotlib.pyplot as plt

#converted directly from: https://github.com/joperea/sw1pers/blob/master/sw1pers_v1_src/SW1PerS_v1.m
#finds the longest (maximum) birth and death time.
def findLongest(matrix):
    if matrix.size == 0:
        return "empty"
    
    most = [] #will contain the coordinates that lasted the longest
    longest = -1
    for m in range(0, len(matrix)): #for each row
        if (matrix[m][1] - matrix[m][0]) > longest:
            #if the birth and death is greater than the longest, label this one most
            most = matrix[m] 
            longest = matrix[m][1] - matrix[m][0]
    return most


#signal should be evenly spaced 
def score(signal, num_cycles, num_points, filePath, plot = True, feature_type = False, allow_trending = False, use_meanshift = False, meanshift_epsilon = 0, use_expaverage = False, expaverage_alpha = 0, use_movingaverage = False, movingaverage_window = 0, use_smoothspline = False, smoothspline_sigma = 0):
    nS = len(signal)
    #signal = np.reshape(signal, 1) #og mathlab code: signal = reshape(signal,1,[]);
    
    #define parameters (Taken from code directly)
    N = 7
    p = 11 
    M = 2*N #embedding dimension
    tau = (2*math.pi) / ((M + 1)*num_cycles)
    

    #moving average
    if(use_expaverage):
        sigLowPass = signal
        for i in range(2, nS):
            sigLowPass[i] = sigLowPass[i-1] + (expaverage_alpha)*(signal(i) - sigLowPass(i-1))
        signal = sigLowPass      

    #detrending - TO ADD FROM SEAN
        
    #create Sliding Window Point Cloud Data
    #this matrix is to embedd our signal onto the point cloud
    t = 2*math.pi*np.linspace(0,1,nS)
    T = (2*math.pi - M*tau)*np.linspace(0,1,num_points)
    #print("T:", T)
    this = np.transpose(np.matrix([*range(0, M+1)]))
    #print("One:", this, type(this))
    t2 = (tau*this)
    #print("Two:",tt)
    #print(tt.shape)
    t3 = np.matmul(t2, np.ones(shape = (1,num_points)))
    #print("Three:",ttt)
    t4 = np.ones((M+1,1))*T
    #print("Four:",tttt)
    t5 = t3 + t4
    #print("Five:",ttttt)
    t6 = np.transpose(t5)
    
    #now we make the point cloud version of the signal
    spline = CubicSpline(t, signal)
    cloud = spline(t6)
    
    #mean center the cloud
    #print(np.mean(signal))
    cloud_centered = cloud - np.mean(signal)
    #print(cloud_centered)
    
    #Normalize (standard deviation)
    step1 = (np.ones((M+1, 1))*math.sqrt(np.sum(np.square(cloud_centered))))
    swCloud = np.divide(cloud_centered, np.transpose(step1))
    SW_Cloud = np.transpose(swCloud)
    #print(SW_Cloud.shape)
    
    #mean shift denoising
    #suggested epsilon is:
    #meanshift_epsilon = 1 - math.cos(math.pi/16) #reccomended in Dr. Perea's code
    #if True:
        #cloudd = np.zeros((SW_Cloud.shape))
        #D = squareform(pdist(SW_Cloud, 'cosine'))
        #print(D, "test")
        #for p in range(0, len(D)):
            #cloud[p] = 
    
    #Now plot the diagram
    if(swCloud.shape[0] > 800):
        diagram = ripser(swCloud, maxdim = 2, metric = "cosine", n_perm = 800)
    else: 
        diagram = ripser(swCloud, maxdim = 2, metric = "cosine")
    dgms = diagram['dgms']
    plot_diagrams(dgms, show = plot)
    #plt.savefig(filePath)
    
    #and to produce the score
    zero = diagram['dgms'][0]
    one = diagram['dgms'][1]  
    two = diagram['dgms'][2]  
    
    scoreH1 = -1 #returns -1 if no change
    scoreH2 = -1#returns -1 if no change
    
    mostPersistantH1 = findLongest(one)
    if mostPersistantH1 != "empty":
        #death minus birth divided sqrt of 3
        scoreH1 = 1 - (mostPersistantH1[1] - mostPersistantH1[0]) / (3**(1/2))
    mostPersistantH2 = findLongest(two)
    if mostPersistantH2 != "empty":
        #death minus birth divided sqrt of 3
        scoreH2 = 1 - (mostPersistantH2[1] - mostPersistantH2[0]) / (3**(1/2))
    
    #SAVE PERSISTANCE DIAGRAM WITH GIVEN PATH
    
    
    return scoreH1, scoreH2

#import pandas as pd
#data = 'hades.csv'
#ts_df = pd.read_csv(data)
#players = ts_df['Players'].tolist()
#(score(players, num_cycles = 50, num_points = len(players), plot = False))

#data = 'katanaZero.csv'
#ts_df = pd.read_csv(data)
#players = ts_df['Players'].tolist()
#print(score(players, num_cycles = 50, num_points = len(players), plot = True))
    
#data = 'katanaZero.csv'
#ts_df = pd.read_csv(data)
#players = ts_df['Players'].tolist()
#print(score(players, num_cycles = 50, num_points = len(players), plot = True))
