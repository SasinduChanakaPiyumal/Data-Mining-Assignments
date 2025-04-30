#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import warnings
warnings.filterwarnings('ignore')


# In[2]:


path = 'task4-dataset.csv'
data = pd.read_csv(path)
df = pd.DataFrame(data)
df.head()


# In[3]:


df.isnull().sum()


# In[4]:


df.info()


# In[5]:


df.describe()


# In[6]:


df['gender'] = df['gender'].astype('category')


# In[7]:


df['likes_pinapple_on_pizza'] = df['likes_pinapple_on_pizza'].astype('category')


# In[8]:


numeric_vals = ['semester','age','height','study',"books_per_year","english_skills",]

for col in numeric_vals:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.tight_layout()
    plt.show()


# In[9]:


categorical_col = ["gender","likes_pinapple_on_pizza","likes_chocolate"]
for col in categorical_col:
    sns.countplot(x=df[col], hue = df[col])
    plt.show()


# In[10]:


sns.boxplot(x=df['gender'], y=df['english_skills'], hue = df['gender'])
plt.show()


# In[11]:


sns.boxplot(x=df['gender'], y=df['height'], hue = df['gender'])
plt.show()


# In[12]:


df = df[df['semester'] <= 40]


# In[13]:


df['height'] = df.groupby('gender')['height'].transform(lambda x: x.fillna(x.median()))


# In[14]:


df['english_skills'] = df['english_skills'].fillna(df['english_skills'].median())


# In[15]:


df['likes_pinapple_on_pizza'] = df['likes_pinapple_on_pizza'].cat.add_categories('Unknown').fillna('Unknown')


# In[16]:


df.isnull().sum()


# In[17]:


corr_matrix = df.corr(numeric_only=True)


# In[18]:


print(corr_matrix)


# In[19]:


sns.pairplot(df, diag_kind='kde', markers='o', palette='husl')

# Show the plot
plt.show()


# In[20]:


plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", cbar=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[21]:


contingency = pd.crosstab(df['gender'], df['likes_chocolate'])

# Plot the stacked bar plot
contingency.plot(kind='bar', stacked=True, figsize=(8, 6), color=['red', 'orange','green'])
plt.title('Stacked Bar Plot: Association Between gender and likes_chocolate')
plt.xlabel('gender')
plt.ylabel('Count')
plt.show()


# In[22]:


plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, cmap='Set3', fmt='d', cbar=False)
plt.title('Heatmap: Association Between Gender and likes_chocolate')
plt.xlabel('likes_chocolate')
plt.ylabel('Gender')
plt.show()


# In[23]:


sns.boxplot(x=df['likes_chocolate'], y=df['age'], hue = df['gender'])
plt.show()


# In[24]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='semester', data=df, marker='o')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Semester')
plt.title('Scatter Plot: Age vs semester')

# Show plot
plt.show()


# In[25]:


df.describe()

