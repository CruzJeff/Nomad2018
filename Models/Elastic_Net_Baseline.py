# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 22:22:39 2018

@author: User
"""


# Importing libraries
import numpy as np
import pandas as pd

# Importing the dataset
train = pd.read_csv("train.csv")
X_test = pd.read_csv("test.csv")
test_id=X_test.id

#Data preprocessing
X_train = train.drop(['id','bandgap_energy_ev','formation_energy_ev_natom'], axis=1)
y_fe = np.log(train['formation_energy_ev_natom']+1)
y_be = np.log(train['bandgap_energy_ev']+1)
X_test = X_test.drop(['id'], axis = 1)

#Create ElasticNet Regression Model
from sklearn.linear_model import ElasticNet
FE_Net = ElasticNet(random_state=42)
BE_Net = ElasticNet(random_state=42)
FE_Net.fit(X_train,y_fe)
BE_Net.fit(X_train,y_be)

# Applying k-Fold Cross Validation
from sklearn.model_selection import cross_val_score
accuracies = cross_val_score(estimator = FE_Net, X = X_train, y = y_fe, cv = 10)
print("FE Mean:",accuracies.mean())
print("FE STD:",accuracies.std())

accuracies = cross_val_score(estimator = BE_Net, X = X_train, y = y_be, cv = 10)
print("BE Mean:",accuracies.mean())
print("BE STD:",accuracies.std())

# Applying Grid Search to find the best parameters
from sklearn.model_selection import GridSearchCV

parameters = [{'alpha': [0.0005,0.005,0.05,0.5,1],
               'l1_ratio': [0.5,0.75,0.9,1],}]

grid_search = GridSearchCV(estimator = FE_Net,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           verbose = 1,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_fe)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Create New Model Based on Results
FE_Net = ElasticNet(alpha=best_parameters['alpha'], #0.0005
                        l1_ratio=best_parameters['l1_ratio'], #0.5
                        random_state=42)

FE_Net.fit(X_train,y_fe)


# Applying Grid Search to find the best parameters
parameters = [{'alpha': [0.0005,0.005,0.05,0.5,1],
               'l1_ratio': [0.5,0.75,0.9,1],}]

grid_search = GridSearchCV(estimator = BE_Net,
                           param_grid = parameters,
                           scoring = 'neg_mean_squared_error',
                           cv = 10,
                           verbose = 1,
                           n_jobs = -1)

grid_search = grid_search.fit(X_train, y_be)
best_accuracy = grid_search.best_score_
best_parameters = grid_search.best_params_

#Create New Model Based on Results
BE_Net = ElasticNet(alpha=best_parameters['alpha'], #0.0005
                        l1_ratio=best_parameters['l1_ratio'], #0.5
                        random_state=42)

BE_Net.fit(X_train,y_be)


#Make Predictions
y_pred_fe = FE_Net.predict(X_test)
y_pred_be = BE_Net.predict(X_test)

import math
for x in range (len(y_pred_fe)):
    y_pred_fe[x] = math.exp(y_pred_fe[x] - 1) 
    y_pred_be[x] = math.exp(y_pred_be[x] - 1) 

#Output Predictions.csv
submission = pd.DataFrame({'formation_energy_ev_natom': y_pred_fe.reshape((y_pred_fe.shape[0])),
                           'bandgap_energy_ev': y_pred_be.reshape((y_pred_be.shape[0])),
                           'Id': test_id})
submission.to_csv("./Submission2.csv", index=False)