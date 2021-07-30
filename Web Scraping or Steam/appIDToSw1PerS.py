# -*- coding: utf-8 -*-
"""
Created on Sat Jul 10 09:14:50 2021

@author: saipr
"""
import pandas as pd
import os
import glob
from datetime import datetime as dt
from pereaVersion import score
import matplotlib.pyplot as plt
  
# use glob to get all the csv files in the folder
path = 'C:\\Users\\saipr\\Downloads\\appID'
csv_files = glob.glob(os.path.join(path, "*.csv"))

appIDList = []
pScores = []
# loop over the list of csv files - first 10 only testing
try:
    for f in csv_files:
        # read the csv file
        print(f)
        df = pd.read_csv(f)
    
        if(df.shape[1] == 3): 
            try:
                df.drop(['Flags'], inplace=True, axis = 1)
            except:
                df.drop(['Twitch Viewers'], inplace=True, axis = 1)
        elif(df.shape[1] == 4): 
                df.drop(['Flags', 'Twitch Viewers'], inplace=True, axis = 1)
    
        df.columns = ['DateTime', 'Players']
        
        df.dropna(inplace = True)
        df['DateTime'] = [dt.strptime(d, "%Y-%m-%d %H:%M:%S") for d in df['DateTime']]
    
        #if the df isn't big enough
        bigEnough = 350
        if(df.shape[0] < bigEnough):
            print("Err TooSmall:", f)
            continue #we'll not finish this df, move to the next one
    
        #now get the persistance score of it, add it to the list
        num_cycles = 50 #Hey how do we pick this value again?
        num_points = df.shape[0]
        signal = df['Players'].tolist()
    
        if(len(signal) < 5):
            print("Err Sig:", f)
            continue
        
        try:
            appID = f.split("\\")[-1] #get the last entry, ie the numbers.csv
            appID = appID.split(".")[0] #get rid of the .csv
            fileName = "C:\\Users\\saipr\\Desktop\\Coding\\Python\\Summer 2021\\diagrams\\"+ appID + ".png"
            pScores.append(score(signal, num_cycles, num_points, filePath = fileName, plot = True))
            appIDList.append(appID)
            plt.savefig(fileName)    
        except:
            print("Err Ripser:", f)
            continue
except:
    pass

#export the list of all the persistance scores
for i in range(0, len(pScores)):
    print("ID:", appIDList[i], " Score:", pScores[i])
print(len(appIDList))
    
dat = pd.DataFrame(list(zip(appIDList, pScores)), columns = ['appID', 'Score'])
dat.to_csv("holyGrail.csv")