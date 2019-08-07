# -*- coding: utf-8 -*-
"""
Created on Tue Aug  6 22:45:23 2019

@author: JigDhwani
"""

import pandas as pd
import matplotlib.pyplot as plt

# Importing the dataset
X = pd.read_csv('train_values.csv', usecols = [1,2,4,5,9,11,13])
X_test = pd.read_csv('test_values.csv', usecols = [1,2,4,5,9,11,13])
y = pd.read_csv('train_labels.csv')
#X['class']=y.iloc[:,0]
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X['thal'] = labelencoder_X.fit_transform(X['thal'])
X_test['thal'] = labelencoder_X.transform(X_test['thal'])

#X.drop(columns = "patient_id", axis = 1, inplace = True)
#X_test.drop(columns = "patient_id", axis = 1, inplace = True)
y.drop(columns = "patient_id", axis = 1, inplace = True)



Correlation=X.corr()

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X, y)
xgb_pred = regressor.predict(X_test)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X,y)
nb_pred = gnb.predict_proba(X_test)
#plt.scatter(X.iloc[:,8],y)
#plt.show()

#from sklearn.decomposition import PCA
#pca = PCA(n_components = 7)
#X = pca.fit_transform(X)
#X_test = pca.transform(X_test)
#explained_variance = pca.explained_variance_ratio_
from sklearn.model_selection import train_test_split
X, X_test, y, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.metrics import log_loss

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X, y)
xgb_preds = regressor.predict(X_test)
xg=log_loss(y_test, xgb_preds, eps=1e-7)
print("XGB: ",xg)

from sklearn.naive_bayes import GaussianNB
gnb = GaussianNB()
gnb.fit(X,y)
nb_preds = gnb.predict_proba(X_test)
nb=log_loss(y_test, nb_preds[:,1])
print("GNB: ",nb)

if nb>xg:
    y_pred = xgb_pred
else:
    y_pred = nb_pred

#from sklearn.model_selection import train_test_split
#X, X_test, y, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)
#
#from sklearn.metrics import mean_squared_error, mean_squared_log_error
#from math import sqrt
#sqrt(mean_squared_error(y_test, y_predtss[:,1]))
#100-mean_squared_log_error(y_test, y_predtss[:,1])

import csv

myData1 = y_pred

myFile = open('submission_test4.csv','w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(myData1)