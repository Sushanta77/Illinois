#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import Lasso,ElasticNet
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder,OneHotEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import xgboost as xgb
import os
import warnings
warnings.filterwarnings("ignore")


# In[4]:


##Please remove the Below code while submitting the code for Grading
y_test = pd.read_csv("y_test.csv")
y_predict_lasso = pd.read_csv("mysubmission1.txt")
y_predict_xgboost = pd.read_csv("mysubmission2.txt")
rmse_lasso = np.sqrt(np.mean((np.log(y_predict_lasso['Sale_Price']) - np.log(y_test['Sale_Price']))**2))
rmse_xgboost = np.sqrt(np.mean((np.log(y_predict_xgboost['Sale_Price']) - np.log(y_test['Sale_Price']))**2))
print ("RMSE Lasso:{:.3f} - RMSE Xgboost:{:.3f}".format(rmse_lasso,rmse_xgboost))

