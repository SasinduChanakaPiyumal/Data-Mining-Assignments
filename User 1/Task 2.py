#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


path = 'task2-dataset.csv'
data = pd.read_csv(path)
df = pd.DataFrame(data)
df.head()


# In[3]:


df.shape


# In[4]:


df.info()


# In[5]:


for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(f'{col} has', df[col].isnull().sum(), 'missing values')
else:
    print('No missing values')


# In[6]:


corr_matrix = df.corr(numeric_only=True)


# In[7]:


print(corr_matrix)


# In[8]:


import pandas as pd
import numpy as np

# Compute the correlation matrix
corr_matrix = df.corr()

# Get the upper triangle of the correlation matrix
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
upper_tri = corr_matrix.where(mask)

# Convert to long format, drop NaNs
corr_pairs = upper_tri.stack().reset_index()
corr_pairs.columns = ['Variable 1', 'Variable 2', 'Correlation']

# Sort by absolute correlation
top_corr_pairs = corr_pairs.reindex(corr_pairs['Correlation'].abs().sort_values(ascending=False).index)

# Show top 20 pairs
print(top_corr_pairs.head(30))

