#!/usr/bin/env python
# coding: utf-8

# In[5]:


import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression,ElasticNet
from sklearn.ensemble import RandomForestRegressor
import xgboost as xgb


# In[6]:


def mypredict(train,test,next_fold,t):
    
    train = pd.concat([train,next_fold])
    X_train = train.copy()
    X_test = test.copy()
    test_pred_ret = test.copy()

    #----------------------------------------------------------
    # Feature Engineering
    #----------------------------------------------------------
    #One Hot Encoding
    X_train['IsHoliday_oh'] = X_train['IsHoliday'].astype(int)
    #Extract Year, Month, Week, Date (Train)
    X_train['day'] = pd.to_datetime(X_train['Date']).dt.day
    X_train['week'] = pd.to_datetime(X_train['Date']).dt.week
    X_train['month'] = pd.to_datetime(X_train['Date']).dt.month
    X_train['year'] = pd.to_datetime(X_train['Date']).dt.year

    y_train = train['Weekly_Sales']
    X_train = X_train.drop(['IsHoliday','Date','Weekly_Sales'],axis=1)

    #One Hot Encoding
    X_test['IsHoliday_oh'] = X_test['IsHoliday'].astype(int)
    #Extract Year, Month, Week, Date (Train)
    X_test['day'] = pd.to_datetime(X_test['Date']).dt.day
    X_test['week'] = pd.to_datetime(X_test['Date']).dt.week
    X_test['month'] = pd.to_datetime(X_test['Date']).dt.month
    X_test['year'] = pd.to_datetime(X_test['Date']).dt.year
    X_test = X_test.drop(['IsHoliday','Date'],axis=1)

    #----------------------------------------------------------
    # Model Building
    #----------------------------------------------------------
    #rf_mod = RandomForestRegressor(n_estimators=150,
    #                               random_state=125247)
    #rf_mod.fit(X_train,y_train)
    
    
    #Random Forest (Prediction)
    #y_preds = rf_mod.predict(X_test)

    #lin_mod = LinearRegression()
    #lin_mod.fit(X_train,y_train)
    
    #Linear Regression (Prediction)
    #y_preds = lin_mod.predict(X_test)
    
    xgb_mod = xgb.XGBRegressor(objective ='reg:squarederror',
                               colsample_bytree = 0.1,
                               learning_rate = 0.04,
                               max_depth = 25,
                               alpha = 1, 
                               n_estimators = 1000
                               )
    
    xgb_mod.fit(X_train,y_train)
    
    #Xgboost (Prediction)
    y_preds = xgb_mod.predict(X_test)

    test_pred_ret['Weekly_Pred'] = y_preds
    
    return train,test_pred_ret
        

