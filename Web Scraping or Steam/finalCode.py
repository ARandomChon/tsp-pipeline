# -*- coding: utf-8 -*-
"""
Created on Wed Jul 21 17:05:59 2021

@author: saipr
"""

from steam_pipeline import steam_pipeline
import pandas as pd
import numpy as np

#import the holy grail
pers = pd.read_csv("holyGrail.csv", index_col=0)

#seperate the APP id into a literal list and get loop scores
apis = pers['appID'].to_list()
scores = [float((x.split(",")[0])[1:]) for x in pers['Score'].to_list()]
pers['scores'] = scores
pers.drop(labels = ['Score'], axis = 1, inplace = True)
pers['appID'] = pers['appID'].astype('string')
#print(pers.head())

#run the pipeline
#dataset = steam_pipeline(apis)
#dataset.to_csv('linData.csv')

#import the pipeline
df = pd.read_csv('linData.csv', index_col = 0)

#turn app id from a float to a string (drop the empty cols)
df.dropna(subset = ['appid'], inplace = True)
df['appid'] = (df['appid'].astype('int32')).astype('string')
df.rename(columns = {'appid':'appID'}, inplace = True)

#find percent positive (pos / (pos + neg)) and add that column
df['percentPos'] = df['positive'] / (df['positive'] + df['negative'])

#turn owners into a categorical - df['col_name'] = df['col_name'].astype('category')
df['owners'] = df['owners'].astype('category')

#find all the words in genres, remove whatever words make no sense, and then keep only the first genre
#then turn it into a categorical
#genres = []
#for g in df['genre']:
    #try:
        #genres.extend(g.split(','))
    #except:
        #continue
#genres = np.array(genres)
#unique = np.unique(genres)
#print(unique)
genres = ['Adventure', 'Casual', 'Massively Multiplayer', 'RPG', 
          'Simulation', 'Sports', 'Strategy', 'Action'] #everything else is other

newGens = []
for g in df['genre']:
    try:
        first = g.split(',')[0]
        if first in genres:
            newGens.append(first)
        else:
            newGens.append("Other")
    except:
        newGens.append("Other")
        continue

df['genre'] = newGens
df['genre'] = df['genre'].astype('category')

#hasMultiplayer column
hasMult = []
for t in df['tags']:
    try:
        if 'Multiplayer' in t:
            hasMult.append(True)
        else:
            hasMult.append(False)
    except:
        hasMult.append(False)
        continue
df['hasMult'] = hasMult
df['hasMult'] = df['hasMult'].astype('bool')

#numLanguages column; drop games without a language?? Games should have language
df.dropna(subset = ['languages'], inplace = True)
df['numLang'] = [len(l.split(',')) for l in df['languages']]           

#go dif between median and average
avMedDiff = []
for i in range(0, df.shape[0]):
    try:
        avMedDiff.append(df['average_forever'][i] - df['median_forever'][i])
    except:
        avMedDiff.append(None)
        continue
df['avMedDiff'] = avMedDiff

#same DevPublisher column
sameDevPub = []
for i in range(0, df.shape[0]):
    try:
        if df['publisher'][i] == df['developer'][i]:
            sameDevPub.append(True)
        else:
            sameDevPub.append(False)
    except:
        sameDevPub.append(False)
        continue
df['sameDevPub'] = sameDevPub
df.to_csv("allInfo.csv") #all possible information\

#keep discount, keep initial price --> find which one is more significant
#keep average forever, median forever --> find which one is more signifcant
#Get rid of all the columns you don't need
df.drop(labels = ['developer', 'publisher', 'score_rank', 'positive',
               'negative', 'userscore', 'average_2weeks', 'median_2weeks',
               'price', 'ccu', 'languages', 'tags', 'average_forever',
               'median_forever'], axis = 1, inplace = True)

#perge persistance to df
merged = pd.merge(pers, df, how = "left", on = "appID")
merged.dropna(inplace = True) #get rid of any missing values
merged.to_csv("merged.csv")

#check for multicolinearity
corrMat = merged.corr()
#Note - average and median playtime have a severe multicolinearity
#print(corrMat) 

#run a linear regression
from statsmodels.formula.api import ols
formula = "scores ~ C(owners) + initialprice + discount + C(genre) + percentPos + C(hasMult) + numLang + avMedDiff + C(sameDevPub) "
mod = ols(formula, merged)
res = mod.fit(use_t=False)
print(res.params)
print(res.pvalues)
print(res.rsquared_adj)


#from sklearn.linear_model import LinearRegression
#from sklearn.model_selection import train_test_split
#data = merged[['owners', 'initialprice', 'discount', 'genre', 'percentPos', 
 #             'hasMult', 'numLang', 'avMedDiff', 'sameDevPub']]

#target = merged['scores']
#X_train, X_test, y_train, y_test = train_test_split(data, target)
#linear_regression = LinearRegression()
#linear_regression.fit(X=X_train, y=y_train)
#predicted = (linear_regression.predict(X_test))
#expected = y_test

#from sklearn import metrics
#metrics.r2_score(expected, predicted)
