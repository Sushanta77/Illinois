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


# In[2]:


def onehot_encoding(x_train_input,x_test_input):
    train_num = x_train_input.shape[0]
    test_num = x_test_input.shape[0]
    df = [x_train_input,x_test_input]
    df_train_test = pd.concat(df)
    
    #Fill the na values to "0" for the feature 'Garage_Yr_Blt'
    #df_train_test['Garage_Yr_Blt'] = df_train_test['Garage_Yr_Blt'].fillna(0)
    
    #Below columns needs to be dropped because of High Imbalance in data
    drop_columns = ['Street', 'Utilities', 'Condition_2', 'Roof_Matl', 'Heating', 
                    'Pool_QC', 'Misc_Feature', 'Low_Qual_Fin_SF', 'Pool_Area', 'Longitude',
                    'Latitude','Land_Slope','Bsmt_Half_Bath','Three_season_porch','Misc_Val',
                    'Garage_Yr_Blt']
    #Let's drop the column
    df_train_test = df_train_test.drop(drop_columns,axis=1)
    
    #Convert the Categorical Variable into dummy variable using Pandas's get_dummmies
    for col_name in df_train_test.columns[df_train_test.dtypes == 'object']:
        df_get_dummies = pd.get_dummies(df_train_test[col_name],drop_first=True,prefix=col_name)
        df_train_test=pd.concat([df_train_test,df_get_dummies],axis=1)
        
    #Drop all the categorical columns, as we have created the dummies columns
    drop_cat_col_name= []
    for col_name in df_train_test.columns[df_train_test.dtypes == 'object']:
        drop_cat_col_name = np.append(drop_cat_col_name,col_name)
    
    df_train_test = df_train_test.drop(drop_cat_col_name,axis=1)
    
    #Split the Train & Test Data
    x_train_return = df_train_test.iloc[0:train_num]
    x_test_return = df_train_test.iloc[train_num:]
    
    return x_train_return,x_test_return


def winsorization(x_train_input,x_test_input):
    #Purposefully, removed the column = "Three_season_porch", "Misc_Val"
    winsorization_cols = ['Lot_Frontage', 'Lot_Area', 'Mas_Vnr_Area', 'BsmtFin_SF_2', 'Bsmt_Unf_SF', 
                          'Total_Bsmt_SF', 'Second_Flr_SF', 'First_Flr_SF', 'Gr_Liv_Area', 'Garage_Area',
                          'Wood_Deck_SF', 'Open_Porch_SF', 'Enclosed_Porch',  
                          'Screen_Porch']
    quan_val = 0.95
    for winso_columns in winsorization_cols:
        col_quant_value = np.quantile(x_train_input[winso_columns],quan_val)
        x_train_input[winso_columns][x_train_input[winso_columns] > col_quant_value] = col_quant_value
        x_test_input[winso_columns][x_test_input[winso_columns] > col_quant_value] = col_quant_value
        #print ("Column : {} 95% Quantile: {}".format(winso_columns,col_quant_value))
        
    return x_train_input,x_test_input

#------------------------------------------------------------------------
#
# Shrinking Methods (Lasso)
#
#------------------------------------------------------------------------
def lasso_model(x_train_lasso,y_train_lasso,x_test_lasso,print_ind=False):
    
    x_train_lasso_PID = x_train_lasso['PID']
    x_test_lasso_PID = x_test_lasso['PID']
    
    x_train_lasso = x_train_lasso.drop(['PID'],axis=1)
    x_test_lasso = x_test_lasso.drop(['PID'],axis=1)

    alpha_ = 0.0001
    
    lasso_model = Lasso(alpha=alpha_)
    lasso_model.fit(x_train_lasso,y_train_lasso)
    y_test_predict  = lasso_model.predict(x_test_lasso)
    df_submission = pd.DataFrame({'PID':x_test_lasso_PID,'Sale_Price':round(np.exp(pd.Series(y_test_predict)),1)})
    
    return df_submission

#------------------------------------------------------------------------
#
# Boosting Model (Xgboost)
#
#------------------------------------------------------------------------
def xgboost_model(x_train_xgboost,y_train_xgboost,x_test_xgboost,print_ind=False):
    max_depth_count_ = 0
    colsample_bytree_ = 0.1
    learning_rate_ = 0.04
    max_depth_ = 25
    alpha_ = 1
    
    x_train_xgboost_PID = x_train_xgboost['PID']
    x_test_xgboost_PID = x_test_xgboost['PID']
    
    x_train_xgboost = x_train_xgboost.drop(['PID'],axis=1)
    x_test_xgboost = x_test_xgboost.drop(['PID'],axis=1)
    
    xg_reg = xgb.XGBRegressor(objective ='reg:squarederror', 
                              colsample_bytree = 0.1, 
                              learning_rate = 0.04,
                              max_depth = 25, 
                              alpha = 1, 
                              n_estimators = 1000)

    xg_reg.fit(x_train_xgboost,y_train_xgboost)
    y_test_predict  = xg_reg.predict(x_test_xgboost)
    df_submission = pd.DataFrame({'PID':x_test_xgboost_PID,'Sale_Price':round(np.exp(pd.Series(y_test_predict)),1)})
    
    return df_submission


# In[3]:


np.random.seed(125247)
train = pd.read_csv("train.csv")
x_test = pd.read_csv("test.csv")

y_train = np.log(train['Sale_Price'])
x_train = train.drop(['Sale_Price'],axis=1)

x_train_onehot,x_test_onehot = onehot_encoding(x_train,x_test)
x_train_final,x_test_final = winsorization(x_train_onehot,x_test_onehot)

#Calling the Model - 1
df_submission_lasso = lasso_model(x_train_final,y_train,x_test_final,True)
#Calling the Model - 2
df_submission_xgboost = xgboost_model(x_train_final,y_train,x_test_final,True)

#Write the Submission File into the Folder
df_submission_lasso.to_csv("mysubmission1.txt",index=False)
df_submission_xgboost.to_csv("mysubmission2.txt",index=False)

