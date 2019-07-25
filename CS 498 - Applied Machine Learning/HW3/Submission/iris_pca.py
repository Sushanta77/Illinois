
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd

def load_data(print_ind=False):
    final_data = []
    final_data.append(np.array(pd.read_csv('data/iris.csv')).T)
    final_data.append(np.array(pd.read_csv('data/dataI.csv')).T)
    final_data.append(np.array(pd.read_csv('data/dataII.csv')).T)
    final_data.append(np.array(pd.read_csv('data/dataIII.csv')).T)
    final_data.append(np.array(pd.read_csv('data/dataIV.csv')).T)
    final_data.append(np.array(pd.read_csv('data/dataV.csv')).T)

    if (print_ind):
        print ("Iris:{} DataI:{} DataII:{} DataIII:{} DataIV:{} DataV:{}".format(iris.shape,dataI.shape,dataII.shape,dataIII.shape,dataIV.shape,dataV.shape))
    return final_data

def data_mean(data,print_ind=False):
    data_mean=np.mean(data,axis=1).reshape(data.shape[0],1)
    if (print_ind):
        print ("Data Mean :{}".format(data_mean.shape))
    return data_mean


def data_final_mean(final_data):
    final_mean_data = []
    for f in range(len(final_data)):
        final_mean_data.append(data_mean(final_data[f],False))
    return final_mean_data


def data_adjusted(data,data_mean,print_ind=False):
    if (print_ind):
        print ("Data Adjusted: {}".format(data.shape))
    return (data-data_mean)
    
    
def data_cov(data,print_ind=False):
    data_cov=np.cov(data,bias=True)
    if (print_ind):
        print ("Data Cov:{}".format(data_cov.shape))
    return data_cov


def data_eigen(data,print_ind=False):
    eigenval,eigenvec = np.linalg.eig(data)
    idx = eigenval.argsort()[::-1]
    eigenval_sort = eigenval[idx]
    eigenvec_sort = eigenvec[:,idx]
    
    if (print_ind):
        print ("Eigen Values:{} Eigen Vectors:{}".format(eigenval_sort.shape,eigenvec_sort.shape))
    return eigenval_sort,eigenvec_sort


def data_final_eigenvec(final_data,final_mean):
    final_eigenvec_data = []
    for f in range(len(final_data)):
        final_eigenvec_data.append(data_eigen(data_cov(data_adjusted(final_data[f],final_mean[f],False),False),False)[1])
    return final_eigenvec_data


def pca_represent_newdataset(data,data_mean,eigenvec,print_ind=False):
    data_transformed = data_adjusted(data,data_mean,False)
    data_hat = np.zeros((data_transformed.shape[0],data_transformed.shape[1]))
    for e in range(eigenvec.shape[1]):
        loop_eigenvec = eigenvec[:,e].reshape(eigenvec[:,e].shape[0],1)
        val = np.dot(loop_eigenvec.T,data_transformed)*loop_eigenvec #Added
        data_hat += val
    data_hat += data_mean
    return data_hat


def data_mse(data1,date2):
    mse = (np.sum((data1-date2)**2))/(data1.shape[1])
    return mse

def generate_number_csv(final_data,final_mean,final_eigenvec):
    final_mse=np.zeros((5,10))
    for row in range(5):
        columns = ['0N','1N','2N','3N','4N','0C','1C','2C','3C','4C']
        df_numbers_temp = pd.DataFrame(columns=columns)
        for col in range(5):
            #Below code is for N Series *****
            if (col == 0):
                new_dataset=pca_represent_newdataset(final_data[row+1],final_mean[0],np.array([[]]))
                mse = data_mse(final_data[0],new_dataset)
                final_mse[row][col] = round(mse,3)
            else:
                new_dataset=pca_represent_newdataset(final_data[row+1],final_mean[0],final_eigenvec[0][:,0:col])
                mse = data_mse(final_data[0],new_dataset)
                final_mse[row][col] = round(mse,3)

            #Below code is for C Series *****
            if (col == 0):
                new_dataset=pca_represent_newdataset(final_data[row+1],final_mean[row+1],np.array([[]]))
                mse = data_mse(final_data[0],new_dataset)
                final_mse[row][col+5] = round(mse,3)
            else:
                new_dataset=pca_represent_newdataset(final_data[row+1],final_mean[row+1],final_eigenvec[row+1][:,0:col])
                mse = data_mse(final_data[0],new_dataset)
                final_mse[row][col+5] = round(mse,3)
    columns = ['0N','1N','2N','3N','4N','0C','1C','2C','3C','4C']
    sub=pd.DataFrame(final_mse,columns=columns)
    sub.to_csv("submission/panda5-numbers.csv",index=False)

    
def generate_recon_csv(final_data,final_mean,final_eigenvec):
    new_dataset=pca_represent_newdataset(final_data[1],final_mean[1],final_eigenvec[1][:,0:2])
    columns = ['Sepal.Length','Sepal.Width','Petal.Length','Petal.Width']
    sub_recon=pd.DataFrame(new_dataset.T,columns=columns)
    sub_recon.to_csv("submission/panda5-recon.csv",index=False)    
    
    
def main():
    final_data = load_data(False)
    final_mean=data_final_mean(final_data)
    final_eigenvec=data_final_eigenvec(final_data,final_mean)
    generate_number_csv(final_data,final_mean,final_eigenvec)
    generate_recon_csv(final_data,final_mean,final_eigenvec)

main()

