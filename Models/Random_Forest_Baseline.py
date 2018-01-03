# -*- coding: utf-8 -*-
"""
Created on Tue Jan  2 21:30:54 2018

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


#Create Basic Random_Forest Model
from sklearn.ensemble import RandomForestRegressor
Random_Forest_fe = RandomForestRegressor(n_estimators = 300, n_jobs=-1, random_state=42)
Random_Forest_be = RandomForestRegressor(n_estimators = 300, n_jobs=-1, random_state=42)
Random_Forest_fe.fit(X_train,y_fe)
Random_Forest_be.fit(X_train,y_be)

#Make Predictions
y_pred_fe = Random_Forest_fe.predict(X_test)
y_pred_be = Random_Forest_be.predict(X_test)

import math
for x in range (len(y_pred_fe)):
    y_pred_fe[x] = math.exp(y_pred_fe[x]) - 1
    y_pred_be[x] = math.exp(y_pred_be[x]) -1

#Output Predictions.csv
submission = pd.DataFrame({'formation_energy_ev_natom': y_pred_fe.reshape((y_pred_fe.shape[0])),
                           'bandgap_energy_ev': y_pred_be.reshape((y_pred_be.shape[0])),
                           'Id': test_id})
submission.to_csv("./Submission.csv", index=False)