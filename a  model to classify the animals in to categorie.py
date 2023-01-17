# -*- coding: utf-8 -*-
"""
Created on Sat Nov  5 19:29:04 2022

@author: Mohd Ariz Khan
"""

# import the data
import pandas as pd
df = pd.read_csv("Zoo.csv")
df.shape
df.head()

# split the data as X and y variable
Y = df["type"]
X = df.iloc[:,0:17]


# EDA (scatter plot between each X and Y)  
import seaborn as sns
sns.set_style(style='darkgrid')
sns.pairplot(df)

# Data Transformation--> one hot encoding
from sklearn.preprocessing import OneHotEncoder
OHE = OneHotEncoder()
X_ohe = OHE.fit_transform(X).toarray()
X_ohe = pd.DataFrame(X_ohe)
X_ohe.shape

# Data Partition
from sklearn.model_selection import train_test_split
X_train, X_test,Y_train, Y_test = train_test_split(X_ohe,Y , test_size=0.3)

# Fit the model
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier(n_neighbors=5, p=2) # k =5 # p=2 --> Eucledian distance
knn.fit(X_train, Y_train)

# Prediction
y_pred_train = knn.predict(X_train)
y_pred_test = knn.predict(X_test)

from sklearn.metrics import accuracy_score
ac1 = accuracy_score(Y_train,y_pred_train)
ac1.round(3)

ac2 = accuracy_score(Y_test,y_pred_test)
ac2.round(3)

#==============================================================
train_accuracy = []
test_accuracy = []    

for i in range(5,20,2):
    knn = KNeighborsClassifier(n_neighbors=i, p=2)
    knn.fit(X_train, Y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_accuracy.append(accuracy_score(Y_train,y_pred_train).round(2))
    test_accuracy.append(accuracy_score(Y_test,y_pred_test).round(2))


d1 = pd.DataFrame(range(5,20,2))
d2 = pd.DataFrame(train_accuracy)
d3 = pd.DataFrame(test_accuracy)
    
pd.concat([d1,d2,d3],axis=1)

#==============================================================
train_accuracy = []
test_accuracy = []    

for i in range(1,500,1):
    X_train, X_test,Y_train, Y_test = train_test_split(X_ohe,Y , test_size=0.3, random_state=i)
    KNeighborsClassifier(n_neighbors=5, p=2)
    knn.fit(X_train, Y_train)
    y_pred_train = knn.predict(X_train)
    y_pred_test = knn.predict(X_test)
    train_accuracy.append(accuracy_score(Y_train,y_pred_train).round(2))
    test_accuracy.append(accuracy_score(Y_test,y_pred_test).round(2))

import numpy as np    
np.mean(train_accuracy).round(3)
np.mean(test_accuracy).round(3)
    
print("Training Accuracy:", np.mean(train_accuracy).round(3)*100)
print("Training Accuracy:", np.mean(test_accuracy).round(3)*100)

#k =5,  83.6,80.0
#k =7,  83.6,80.0
#k =9,  83.6,80.0
#k =11, 83.6,80.0
#from plots at k=5 we get best model
#model building at k=5 