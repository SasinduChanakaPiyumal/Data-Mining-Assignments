#!/usr/bin/env python
# coding: utf-8

# In[30]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[31]:


path = 'task4-dataset.csv'
data = pd.read_csv(path)
df = pd.DataFrame(data)
df.head()


# In[32]:


df.info()


# In[33]:


df.isnull().sum()


# In[34]:


df.describe()


# In[35]:


import seaborn as sns


# In[36]:


df['gender'] = df['gender'].astype('category')


# In[37]:


df['likes_pinapple_on_pizza'] = df['likes_pinapple_on_pizza'].astype('category')


# In[38]:


numeric_vals = ['semester','height','age','study',"english_skills","books_per_year"]

for col in numeric_vals:
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    sns.histplot(df[col], kde=True)
    plt.subplot(1, 2, 2)
    sns.boxplot(x=df[col])
    plt.tight_layout()
    plt.show()


# In[39]:


sns.boxplot(x=df['gender'], y=df['height'], hue = df['gender'])
plt.show()


# In[40]:


categorical_col = ["gender","likes_pinapple_on_pizza","likes_chocolate"]
for col in categorical_col:
    sns.countplot(x=df[col], hue = df[col])
    plt.show()


# In[41]:


sns.boxplot(x=df['gender'], y=df['english_skills'], hue = df['gender'])
plt.show()


# In[42]:


df = df[df['semester'] <= 40]


# In[43]:


df['height'] = df.groupby('gender')['height'].transform(lambda x: x.fillna(x.median()))


# In[44]:


df['english_skills'] = df['english_skills'].fillna(df['english_skills'].median())


# In[45]:


df['likes_pinapple_on_pizza'] = df['likes_pinapple_on_pizza'].cat.add_categories('Unknown').fillna('Unknown')


# In[46]:


df.isnull().sum()


# In[47]:


corr_matrix = df.corr(numeric_only=True)


# In[48]:


print(corr_matrix)


# In[49]:


plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', linewidths=0.5, fmt=".2f", cbar=True)
plt.title('Correlation Matrix Heatmap')
plt.show()


# In[50]:


sns.pairplot(df, diag_kind='kde', markers='o', palette='coolwarm')

# Show the plot
plt.show()


# In[51]:


contingency = pd.crosstab(df['gender'], df['likes_chocolate'])

# Plot the stacked bar plot
contingency.plot(kind='bar', stacked=True, figsize=(8, 6), color=['skyblue', 'orange','green'])
plt.title('Stacked Bar Plot: Association Between gender and likes_chocolate')
plt.xlabel('gender')
plt.ylabel('Count')
plt.show()


# In[52]:


plt.figure(figsize=(8, 6))
sns.heatmap(contingency, annot=True, cmap='Blues', fmt='d', cbar=False)
plt.title('Heatmap: Association Between feature_1 and feature_2')
plt.xlabel('likes_chocolate')
plt.ylabel('Gender')
plt.show()


# In[53]:


from statsmodels.graphics.mosaicplot import mosaic


# Create a mosaic plot
mosaic(df, ['gender', 'likes_chocolate'])
plt.title('Mosaic Plot: Association Between feature_1 and feature_2')
plt.show()


# In[54]:


sns.boxplot(x=df['likes_chocolate'], y=df['age'], hue = df['gender'])
plt.show()


# In[55]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='semester', data=df, color='green', marker='o')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('Semester')
plt.title('Scatter Plot: Vehicle Count vs Reported Fatalities')

# Show plot
plt.show()


# In[56]:


plt.figure(figsize=(8, 6))
sns.scatterplot(x='age', y='height', data=df, color='blue', marker='o')

# Add labels and title
plt.xlabel('Age')
plt.ylabel('height')
plt.title('Scatter Plot: Vehicle Count vs Reported Fatalities')

# Show plot
plt.show()


# In[57]:


df.describe()


# In[ ]:




