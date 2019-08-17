# -*- coding: utf-8 -*-
"""
Created on Fri Apr  5 17:42:46 2019

@author: JigDhwani
"""

import numpy as np
import pandas as pd

X_train = pd.read_csv('train_values.csv')
X_train = X_train.iloc[:,1:]
y_train = pd.read_csv('train_labels.csv')
y_train = y_train.iloc[:,1:]
X_test = pd.read_csv('test_values.csv')
X_test = X_test.iloc[:,1:]

##Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
#X_train[:,[0,1,3,4,5,6,9,12]] = labelencoder_X.fit_transform(X_train[:,[0,1,3,4,5,6,9,12]])
X_train[['thal']] = labelencoder_X.fit_transform(X_train['thal'])
X_test[['thal']] = labelencoder_X.fit_transform(X_test['thal'])
#onehotencoder = OneHotEncoder(categorical_features = [[0,1,3,4,5,6,8,11]])
#X_train = onehotencoder.fit_transform(X_train).toarray()
#X_test = onehotencoder.transform(X_test).toarray()
# Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1), copy=True)
X_train.iloc[:,[2,7,10,11]]=sc.fit_transform(X_train.iloc[:,[2,7,10,11]])
X_test.iloc[:,[2,7,10,11]]=sc.transform(X_test.iloc[:,[2,7,10,11]])

#
from sklearn.decomposition import PCA
pca = PCA(n_components = None)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

#X_train = pd.DataFrame(X_train)
#X_test = pd.DataFrame(X_test)
#from sklearn.model_selection import train_test_split
#X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size = 0.2, random_state = 0)
from sklearn.ensemble import RandomForestRegressor
regressor = RandomForestRegressor(n_estimators=257,random_state=0)
regressor.fit(X_train, y_train)
y_pred=regressor.predict(X_test)

from xgboost import XGBRegressor
xgb = XGBRegressor(max_depth=25)

xgb.fit(X_train, y_train)
y_pred = xgb.predict(X_test)
#from sklearn.preprocessing import MinMaxScaler
#sc = MinMaxScaler(feature_range=(0, 1), copy=False)
#y_pred=y_pred.reshape((-1,1))
#y_pred=sc.fit_transform(y_pred)
#y_pred = pd.DataFrame(y_pred)
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#for i,k in enumerate(y_pred):
#    if k>0.5:
#        y_pred[i]=np.float64(1)
#    else:
#        y_pred[i]=np.float64(0)
import csv
myData1 = y_pred
#output_array = np.array(myData)
#np.savetxt("my_output_file.csv", output_array, delimiter=",")
myFile = open('submission_test.csv','w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(myData1)
#    myFile.close()