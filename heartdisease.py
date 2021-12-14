# -*- coding: utf-8 -*-
"""
Created on Fri Feb  7 15:05:29 2020

@author: Jegan
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#importing dataset
dataset=pd.read_csv('Training_dataset.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,:])
x[:,:]=imputer.transform(x[:,:])
x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.75,random_state=0)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X=LabelEncoder()
x[:,1]=labelencoder_X.fit_transform(x[:,1])
onehotencoder=OneHotEncoder(categorical_features=[1])
x=onehotencoder.fit_transform(x).toarray()
labelencoder_y=LabelEncoder()
y=labelencoder_y.fit_transform(y)

print(x)