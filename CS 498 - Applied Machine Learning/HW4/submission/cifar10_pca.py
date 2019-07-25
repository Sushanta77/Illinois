
# coding: utf-8

# In[53]:


import numpy as np
import pandas as pd
from sklearn.decomposition import pca
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')
get_ipython().run_line_magic('matplotlib', 'inline')


def unpickle(file):
    import pickle
    with open(file, 'rb') as fo:
        dict = pickle.load(fo, encoding='bytes')
    return dict


def load_dataset(Print_Ind=True):
    filename=['data_batch_1','data_batch_2','data_batch_3','data_batch_4','data_batch_5','test_batch']
    cifar_10=np.zeros((len(filename)*10000,3072))
    cifar_10_per_class = []
    labels=np.zeros((len(filename)*10000,1))
    start=0
    end=10000
    for i in range(len(filename)):
        dict=unpickle('data/'+filename[i])
        cifar_10[start:end,:]=np.asarray(dict[b'data'])
        labels[start:end,:]=np.array(dict[b'labels']).reshape(len(dict[b'labels']),1)
        start = start+10000
        end = end+10000
    unique_labels=np.unique(labels)
    for u in unique_labels:
        index_per_class,_ = np.where(labels == u)
        cifar_10_per_class.append(cifar_10[index_per_class].T)
    if (Print_Ind):
        print ("Cifar Per Class:{}  Labels:{}".format(len(cifar_10_per_class),len(labels)))
    return cifar_10_per_class,labels


def plt_img(img):
    img_transformed = img.reshape(3,32,32).transpose([1, 2, 0])
    plt.imshow(img_transformed)


def mean_per_class(data,Print_Ind=True):
    cifar_10_mean_per_class = []
    for u in range(len(data)):
        cifar_10_mean_per_class.append(np.mean(data[u],axis=1).reshape(3072,1))
    if (Print_Ind):
        print ("Cifar 10 Per Class: {}".format(len(cifar_10_mean_per_class)))
    return cifar_10_mean_per_class


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
        print ("Eigen Values:{} Eigen Vectors:{}".format(len(eigenval_sort),len(eigenvec_sort)))
    return eigenval_sort,eigenvec_sort


def data_final_eigenvec(final_data,final_mean,Print_Ind=True):
    final_eigenvec_data = []
    for f in range(len(final_data)):
        final_eigenvec_data.append(data_eigen(data_cov(data_adjusted(final_data[f],final_mean[f],False),False),False)[1])
    if (Print_Ind):
        print ("Final EigenVec:{}".format(len(final_eigenvec_data)))
    return final_eigenvec_data


def pca_represent_newdataset(data,data_mean,eigenvec,add_mean_Ind=True,print_ind=False):
    data_transformed = data_adjusted(data,data_mean,False)
    data_hat = np.zeros((data_transformed.shape[0],data_transformed.shape[1]))
    for e in range(eigenvec.shape[1]):
        loop_eigenvec = eigenvec[:,e].reshape(eigenvec[:,e].shape[0],1)
        val = np.dot(loop_eigenvec.T,data_transformed)*loop_eigenvec #Added
        data_hat += val
    if (add_mean_Ind):
        data_hat += data_mean
    return data_hat


def data_mse(data1,date2):
    mse = (np.sum((data1-date2)**2))/(data1.shape[1])
    return mse


def error_each_class(cifar_10_per_class,mean_cifar_10_per_class,eigenvec_cifar_10_per_class,Print_Ind=True):
    mse_each_class = []
    for c in range(len(cifar_10_per_class)):
        x_hat = pca_represent_newdataset(cifar_10_per_class[c],mean_cifar_10_per_class[c],eigenvec_cifar_10_per_class[c][:,0:20])
        mse_each_class.append(data_mse(cifar_10_per_class[c],x_hat))
    if (Print_Ind):
        print ("MSE Per Class: {}".format(len(mse_each_class)))
    return mse_each_class


def plot_error(mse_each_class,bar_ind=False):
    label=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    fig,ax = plt.subplots(figsize=(9,6))
    if (bar_ind):
        ax.bar(np.arange(len(label)),mse_each_class)
        plt.xticks(np.arange(len(label)),label)
    else:
        ax.scatter(x=np.arange(10),y=mse_each_class)
        for i,txt in enumerate(label):
            ax.annotate(txt,(i,mse_each_class[i]))
    plt.xlabel('Category')
    plt.ylabel('MSE')
    plt.title('Category Error')
    plt.show()

    
def euclidean_distance(mean_cifar_10_per_class):
    euclidean_distance_per_class=np.zeros((10,10))
    for m in range(len(mean_cifar_10_per_class)):
        for k in range(len(mean_cifar_10_per_class)):
            euclidean_distance_per_class[m][k] = np.sum((mean_cifar_10_per_class[m] - mean_cifar_10_per_class[k])**2)
    np.savetxt("submission/partb_distances.csv",euclidean_distance_per_class,delimiter=",")    
    return euclidean_distance_per_class


def part_c(cifar_10_per_class,mean_cifar_10_per_class,eigenvec_cifar_10_per_class,print_Ind=False):
    euclidean_distance_part_c=np.zeros((10,10))
    i = 0
    for m in range(len(mean_cifar_10_per_class)):
        for k in range(len(mean_cifar_10_per_class)):
            i+=1
            x_hat_A_B=pca_represent_newdataset(cifar_10_per_class[m],mean_cifar_10_per_class[m],eigenvec_cifar_10_per_class[k][:,0:20])
            compute_E_A_B=data_mse(cifar_10_per_class[m],x_hat_A_B)
            x_hat_B_A=pca_represent_newdataset(cifar_10_per_class[k],mean_cifar_10_per_class[k],eigenvec_cifar_10_per_class[m][:,0:20])
            compute_E_B_A=data_mse(cifar_10_per_class[k],x_hat_B_A)
            X=((compute_E_A_B + compute_E_B_A) / 2)
            if (print_Ind):
                print ("{} Compute i:{} j:{} A_B:{}  B_A:{}".format(i,m,k,compute_E_A_B,compute_E_B_A))
            euclidean_distance_part_c[m][k]=X
    np.savetxt("submission/partc_distances.csv",euclidean_distance_part_c,delimiter=",")
    return euclidean_distance_part_c    
    
def multi_dimensional_scaling(data,n_components=2,PrintInd=True):
    N=data.shape[1]
    X_1=np.ones((N,1))
    A=np.identity(N)-(np.dot(X_1,X_1.T))/10
    W = (-1/2)*np.dot(np.dot(A,data),(A.T))
    W_eigenval,W_eigenvec=data_eigen(W)
    u=W_eigenvec[:,0:n_components]
    lam=W_eigenval[0:n_components]
    lam_sqrt = np.sqrt(lam)
    Y = u*lam_sqrt
    return Y


def plot_principal_coordinate_analysis(Y,title):
    label=['airplane','automobile','bird','cat','deer','dog','frog','horse','ship','truck']
    fig,ax=plt.subplots()
    plt.scatter(Y[:,0],Y[:,1])
    plt.xlabel('Component 1')
    plt.ylabel('Component 2')
    plt.title(title)
    for i,text in enumerate(label):
        ax.annotate(text,(Y[i,0],Y[i,1]))
    
def main ():
    cifar_10_per_class,labels,=load_dataset()
    mean_cifar_10_per_class=mean_per_class(cifar_10_per_class)
    eigenvec_cifar_10_per_class=data_final_eigenvec(cifar_10_per_class,mean_cifar_10_per_class)
    plt_img(mean_cifar_10_per_class[0].astype('int'))
    plt_img(mean_cifar_10_per_class[1].astype('int'))
    plt_img(mean_cifar_10_per_class[2].astype('int'))
    plt_img(mean_cifar_10_per_class[3].astype('int'))
    plt_img(mean_cifar_10_per_class[4].astype('int'))
    plt_img(mean_cifar_10_per_class[5].astype('int'))
    plt_img(mean_cifar_10_per_class[6].astype('int'))
    plt_img(mean_cifar_10_per_class[7].astype('int'))
    plt_img(mean_cifar_10_per_class[8].astype('int'))
    plt_img(mean_cifar_10_per_class[9].astype('int'))
    mse_each_class=error_each_class(cifar_10_per_class,mean_cifar_10_per_class,eigenvec_cifar_10_per_class)
    plot_error(mse_each_class,False)
    plot_error(mse_each_class,True)
    #euclidean_distance(mean_cifar_10_per_class)
    euclidean_distance_per_class=euclidean_distance(mean_cifar_10_per_class)
    Y = multi_dimensional_scaling(euclidean_distance_per_class,n_components=2,PrintInd=True)
    plot_principal_coordinate_analysis(Y,'Part B')

    euclidean_distance_part_c = euclidean_distance_part_c(cifar_10_per_class,mean_cifar_10_per_class,eigenvec_cifar_10_per_class)
    #part_c=np.array(pd.read_csv("submission/partc_distances.csv",header=None))
    Y = multi_dimensional_scaling(euclidean_distance_part_c,n_components=2,PrintInd=True)
    plot_principal_coordinate_analysis(Y,'Part C')
    
#main()


# In[3]:


cifar_10_per_class,labels,=load_dataset()
mean_cifar_10_per_class=mean_per_class(cifar_10_per_class)
eigenvec_cifar_10_per_class=data_final_eigenvec(cifar_10_per_class,mean_cifar_10_per_class)
euclidean_distance_per_class=euclidean_distance(mean_cifar_10_per_class)


# In[34]:


mse_each_class=error_each_class(cifar_10_per_class,mean_cifar_10_per_class,eigenvec_cifar_10_per_class)


# In[52]:


plot_error(mse_each_class,True)


# In[55]:


euclidean_distance_per_class=euclidean_distance(mean_cifar_10_per_class)
Y = multi_dimensional_scaling(euclidean_distance_per_class,n_components=2,PrintInd=True)
plot_principal_coordinate_analysis(Y,'Part B')


# In[56]:


part_c=np.array(pd.read_csv("submission/partc_distances.csv",header=None))
Y = multi_dimensional_scaling(part_c,n_components=2,PrintInd=True)
plot_principal_coordinate_analysis(Y,'Part C')

