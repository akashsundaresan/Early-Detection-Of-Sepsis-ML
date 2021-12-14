# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split

#importing dataset
dataset=pd.read_csv('Training_dataset.csv')

x=dataset.iloc[:,:-1].values
y=dataset.iloc[:,-1].values

#from sklearn.preprocessing import Imputer
imputer = Imputer(missing_values='NaN',strategy='mean',axis=0)
imputer=imputer.fit(x[:,:])
x[:,:]=imputer.transform(x[:,:])
#x_train,x_test,y_train, y_test=train_test_split(x,y,test_size=0.75,random_state=0)



#from sklearn.preprocessing import StandardScaler
#sc_X = StandardScaler()
#x_train=sc_X.fit_transform(x_train)
#x_test=sc_X.transform(x_test)



#dataset = read_csv('pima-indians-diabetes.csv', header=None)
# mark zero values as missing or NaN
#dataset[[1,2,3,4,5]] = dataset[[1,2,3,4,5]].replace(0, numpy.NaN)
# drop rows with missing values
#dataset.dropna(inplace=True)
# summarize the number of rows and columns in the dataset
print(dataset.shape)