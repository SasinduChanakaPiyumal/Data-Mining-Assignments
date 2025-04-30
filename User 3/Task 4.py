#!/usr/bin/env python
# coding: utf-8

# In[28]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[29]:


path = 'task4-dataset.csv'
data = pd.read_csv(path)
df = pd.DataFrame(data)
df.head()


# In[30]:


df.info()


# In[31]:


df.isnull().sum()


# In[32]:


df.describe()


# In[33]:


df['gender'] = df['gender'].astype('category')


# In[34]:


df['likes_pinapple_on_pizza'] = df['likes_pinapple_on_pizza'].astype('category')


# In[35]:


numeric_vals = ['semester','height','age','study',"english_skills","books_per_year"]

for col in numeric_vals:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True,color='skyblue')
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col],color='skyblue')
    plt.tight_layout()
    plt.show()


# In[36]:


categorical_col = ["gender","likes_pinapple_on_pizza","likes_chocolate"]
for col in categorical_col:
    sns.countplot(x=df[col], hue = df[col])
    plt.show()


# In[37]:


sns.boxplot(x=df['gender'], y=df['english_skills'], hue = df['gender'])
plt.show()


# In[38]:


sns.boxplot(x=df['gender'], y=df['height'], hue = df['gender'])
plt.show()


# In[39]:


df = df[df['semester'] <= 40]


# In[40]:


df['height'] = df.groupby('gender')['height'].transform(lambda x: x.fillna(x.median()))


# In[41]:


df['likes_pinapple_on_pizza'] = df['likes_pinapple_on_pizza'].cat.add_categories('Unknown').fillna('Unknown')


# In[42]:


df['english_skills'] = df['english_skills'].fillna(df['english_skills'].median())


# In[43]:


df.isnull().sum()


# In[44]:


corr_matrix = df.corr(numeric_only=True)


# In[45]:


print(corr_matrix)


# In[46]:


sns.pairplot(df, diag_kind='kde', markers='o', palette='husl')

# Show the plot
plt.show()


# In[47]:


plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='Set2', linewidths=0.5, fmt=".2f", cbar=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[48]:


contingency = pd.crosstab(df['gender'], df['likes_chocolate'])

# Plot the stacked bar plot
contingency.plot(kind='bar', stacked=True, figsize=(8, 6), color=['red', 'orange','green'])
plt.title('Stacked Bar Plot: Association Between gender and likes_chocolate')
plt.xlabel('gender')
plt.ylabel('Count')
plt.show()


# In[49]:


plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Heatmap: Association Between Gender and likes_chocolate')
plt.xlabel('likes_chocolate')
plt.ylabel('Gender')
plt.show()


# In[50]:


sns.boxplot(x=df['likes_chocolate'], y=df['age'], hue = df['gender'])
plt.show()


# In[51]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='semester', data=df, marker='o')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Semester')
plt.title('Scatter Plot: Age vs semester')

# Show plot
plt.show()


# In[52]:


df.describe()


# In[ ]:




