#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 16 21:04:01 2019

@author: yiransun
"""

import h5py
from random import shuffle, randint

import numpy as np
import pandas as pd
import seaborn as sns

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Reshape, GlobalAveragePooling1D
from keras.layers import Conv2D, MaxPooling2D, Conv1D, MaxPooling1D
from keras.utils import np_utils

def process_data():
    filename='/Users/yiransun/Desktop/OpportunityUCIDataset/opportunity.h5'
    f=h5py.File(filename,'r')
    
    X_train=np.asarray(f['train']['inputs'])
    Y_train=np.asarray(f['train']['targets'])
    X_test=np.asarray(f['test']['inputs'])
    Y_test=np.asarray(f['test']['targets'])
    
    X=np.concatenate((X_train,X_test))
    Y=np.concatenate((Y_train,Y_test))
    
    time_period=20
    n=X.shape[0]
    n_sensors=X.shape[1]
    q=n//time_period
    X=X[:q*time_period]
    X=X.reshape((q, time_period, n_sensors))
    
    y=np.zeros((q,))
    y_p=np.zeros((q,)) #the private hypothesis
    for i in range(q):
        if(np.any(Y[i*time_period:(i+1)*time_period]==6)):
            y[i]=1
        if(np.any(Y[i*time_period:(i+1)*time_period]==7)):
            y_p[i]=1
    
    
    inputs=[]
    for i in range(X.shape[0]):
        if y[i]==1 or y_p[i]==1:
            inputs.append([])
            inputs[-1].extend([X[i],y[i],y_p[i]])
        else:
            a=randint(1,40)
            if(a==5):
               inputs.append([])
               inputs[-1].extend([X[i],y[i],y_p[i]]) 
    
    shuffle(inputs)
    targets_p=np.asarray([x[-1] for x in inputs])
    targets=np.asarray([x[-2] for x in inputs])
    inputs=np.asarray([x[0] for x in inputs])
    
    n_train=len(inputs)*4//5
    X_train=np.asarray(inputs[:n_train])
    y_train=np.asarray(targets[:n_train])
    y_p_train=np.asarray(targets_p[:n_train])
    
    X_test=np.asarray(inputs[n_train:])
    y_test=np.asarray(targets[n_train:])
    y_p_test=np.asarray(targets_p[n_train:])
    
    return (X_train,y_train,y_p_train,X_test,y_test,y_p_test)


(X_train,y_train,y_p_train,X_test,y_test,y_p_test)=process_data()

model=Sequential()
model.add(Conv1D(50,4,activation='relu', input_shape=(time_period,n_sensors)))
model.add(Conv1D(40,3,activation='relu'))
model.add(MaxPooling1D(2))
model.add(Conv1D(30,3,activation='relu'))
model.add(Conv1D(20,3,activation='relu'))
model.add(MaxPooling1D(3))
model.add(Dropout(0.5))
model.add(Flatten())
model.add(Dense(10,activation='relu'))
model.add(Dense(1,activation='sigmoid'))

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['accuracy'])
model.fit(X_train,y_train,batch_size=40, epochs=20)
score=model.evaluate(x=X_test,y=y_test,batch_size=20)



