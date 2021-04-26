#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ttest_ind
import warnings 
warnings.filterwarnings('ignore')


# In[2]:


def prep_train_test(x_train, x_test):
    
    #Remove the HTML tags from the review
    x_train["review"] = x_train["review"].str.replace('<br /><br />',' ')
    x_test ["review"] = x_test["review"].str.replace('<br /><br />',' ')

    #Defined the Stopwords
    stop_words=['the','with','he','she','also','made','had','out','in','his','hers','there','was','then'] 
    
    #Tfidf Vectorization
    cv = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2), min_df=20, max_df=0.3)
    X_train = cv.fit_transform(train["review"]).toarray()
    X_test  = cv.transform(test["review"]).toarray()
    
    #Extract the Vocabulary Size
    vocab = np.array(cv.get_feature_names())

    return X_train,X_test, vocab


# In[3]:


#Load the Vocab
myvocab_list = np.loadtxt('myvocab.txt', dtype=np.str, delimiter='\n')

#Load the Train, Test, Test_y data
train = pd.read_csv("train.tsv",sep = "\t")
test = pd.read_csv("test.tsv", sep = "\t")

#Retrieve the y_Train and y_test
y_train = train["sentiment"]

#Remove the Stopwords and TFIDF Vectorization
X_train,X_test,vocab = prep_train_test(train.copy(), test.copy())
indices = np.where(np.in1d(vocab, myvocab_list))[0]
X_train = X_train[:, indices].copy()
X_test  = X_test [:, indices].copy()
    
#Logistic Model Building
model = LogisticRegression(penalty='l2',C=17, random_state=125247)
_ = model.fit(X_train, y_train)
probs = model.predict_proba(X_test)[:,1]

df_probs = pd.DataFrame(probs)
df_probs = df_probs.rename(columns={0:'prob'})

mysubmission = pd.concat([test[["id"]], df_probs.reindex(test[["id"]].index)], axis=1)

mysubmission.to_csv("mysubmission.txt",sep="\t",index=False)

