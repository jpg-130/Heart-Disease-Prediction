# -*- coding: utf-8 -*-
"""
Created on Tue Mar 12 10:38:49 2019

@author: JigDhwani
"""

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# Importing the dataset
X = pd.read_csv('train_values.csv')
X_test = pd.read_csv('test_values.csv')
y = pd.read_csv('train_labels.csv')

#X['class']=y['heart_disease_present']
# Encoding categorical data
# Encoding the Independent Variable
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder_X = LabelEncoder()
X['thal'] = labelencoder_X.fit_transform(X['thal'])
X_test['thal'] = labelencoder_X.transform(X_test['thal'])
#X_test['thal'] = labelencoder_X.fit_transform(X.thal.values)

X.drop(columns = "patient_id", axis = 1, inplace = True)
X_test.drop(columns = "patient_id", axis = 1, inplace = True)
y.drop(columns = "patient_id", axis = 1, inplace = True)

Correlation=X.corr()

from xgboost import XGBRegressor
regressor = XGBRegressor()
regressor.fit(X, y)
y_predtest1= regressor.predict(X_test)

#onehotencoder = OneHotEncoder(categorical_features = [1])
#X_new= np.reshape(X_new, (180, 1))
#X_new = onehotencoder.fit_transform(X.thal.values).toarray()
## Encoding the Dependent Variable
#labelencoder_y = LabelEncoder()
#y = labelencoder_y.fit_transform(y)

##Splitting the dataset into the Training set and Test set
from sklearn.model_selection import train_test_split
X, X_test, y, y_test = train_test_split(X, y, test_size = 0.1, random_state = 0)

from sklearn.metrics import mean_squared_error, mean_squared_log_error
from math import sqrt
sqrt(mean_squared_error(y_test, y_predtest1))
100-mean_squared_log_error(y_test, y_predtest1)
# Applying PCA
from sklearn.decomposition import PCA
pca = PCA(n_components = 4)
X_train = pca.fit_transform(X_train)
X_test = pca.transform(X_test)
explained_variance = pca.explained_variance_ratio_

explained_variance=np.reshape(explained_variance, (4,1))

#from sklearn.svm import SVM
#regressor = SVM()
#regressor.fit(X_train,y_train)
#y_predsvm= regressor.predict(X_test)

#y_pred[y_pred < 0]= float(0)
# Importing the Keras libraries and packages
import keras
from keras.models import Sequential
from keras.layers import Dense

# Initialising the ANN
classifier = Sequential()

# Adding the input layer and the first hidden layer
classifier.add(Dense(output_dim = 11, init = 'uniform', activation = 'relu', input_dim = 4))

# Adding the second hidden layer
classifier.add(Dense(output_dim = 4, init = 'uniform', activation = 'relu'))

# Adding the output layer
classifier.add(Dense(output_dim = 1, init = 'uniform', activation = 'sigmoid'))

# Compiling the ANN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

# Fitting the ANN to the Training set
classifier.fit(X_train, y_train, batch_size = 10, nb_epoch = 50)

# Part 3 - Making the predictions and evaluating the model

# Predicting the Test set results
y_pred2 = classifier.predict(X_test)
#y_pred = (y_pred > 0.5)
#y_pred=y_pred.astype(float)
# Making the Confusion Matrix
#from sklearn.metrics import confusion_matrix
#cm = confusion_matrix(y_test, y_pred)
#submission file code
import csv

myData = y_predtest

myFile = open('submission_test1.csv','w')
with myFile:
    writer = csv.writer(myFile)
    writer.writerow(myData)