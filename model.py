#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jan 26 18:37:46 2020

@author: abish
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle

dataset = pd.read_csv('hiring.csv')
dataset['experience'].fillna(0,inplace =True)
dataset['test_score'].fillna(dataset['test_score'].mean(),inplace =True)

X = dataset.iloc[:,:3]

def convert_to_int(word):
    word_dict ={0:0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7
                ,'eight':8,'nine':9,'ten':10,'eleven':11
                }

    return word_dict[word]

X['experience'] =X['experience'].apply(lambda x: convert_to_int(x))


Y = dataset.iloc[:,-1]


from sklearn.linear_model import LinearRegression

regressor = LinearRegression()
regressor.fit(X,Y)

pickle.dump(regressor,open('model.pk1 ','wb'))

model =pickle.load(open('model.pk1','rb'))

print(model.predict([ [ 4,5,6] ]))