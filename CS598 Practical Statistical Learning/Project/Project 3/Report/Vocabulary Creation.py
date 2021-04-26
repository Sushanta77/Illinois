#!/usr/bin/env python
# coding: utf-8

# ### Load Python Packages required to run this program

# In[1]:


import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from scipy.stats import ttest_ind
import warnings 
warnings.filterwarnings('ignore')


# ### Load the all Data TSV file and Extract the "Sentiment" column

# In[2]:


#Load the Train, Test, Test_y data
x_data = pd.read_csv("../data/alldata.tsv",sep="\t")
y_train = x_data["sentiment"]


# ### Do the following trasnformation in the "review" text
# (1) Remove all the HTML tags <br /><br />
# 
# (2) Using the TErm Frequency (TF) and Inverted Document Frequency (IDF) to Vectorized the Text. In the same time, remove the stopwords which is defined in the stop_words list
# 
# (3) Convert the Input data into the transformed Tfidf Vectorized data
# 

# In[3]:


x_data["review"] = x_data["review"].str.replace('<br /><br />',' ')
#Define the stopwords
stop_words=['the','with','he','she','also','made','had','out','in','his','hers','there','was','then'] 

#Tfidf Vectorizer
cv = TfidfVectorizer(stop_words=stop_words, ngram_range=(1,2), min_df=20, max_df=0.3)
#Transform the Data in Tfidf Vector
X_data = cv.fit_transform(x_data["review"]).toarray()
#Get the Vocablist
vocab = np.array(cv.get_feature_names())


# ### Use the ttest_ind from the scipy packages to calculate the  T-test for the means of two independent samples
# 
# (1) Generate & Store the tstat score against each word retrieved from the above code
# 
# (2) Generate & Store the tstat score with absolute value for each word retrieved from the above code
# 
# (3) Sort the value based on the magn_tstat in descending order
# 
# (4) Pick the first 1999 vocab from the list based on the magn_tstat values
# 
# 

# In[4]:


t_test = ttest_ind(X_data[y_train==1, :], X_data[y_train==0, :])

voc_df = pd.DataFrame({'tstat': t_test.statistic, 'word': vocab})
voc_df['magn_tstat'] = voc_df.tstat.abs()
voc_df = voc_df.sort_values('magn_tstat',ascending=False)


voc_df = voc_df.head(1999)
#voc_df['weight'] = np.power((voc_df.magn_tstat - voc_df.magn_tstat.min()), 1.2)
#voc_df['weight'] = (voc_df['weight'] / voc_df.weight.max() * 21 * np.sign(voc_df.tstat)).round(4)

#voc_df[['word','weight']].to_csv('../data/word_weights.csv', index=False)
np.savetxt('../data/myvocab.txt',voc_df.word.values, fmt='%s')

