#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


path = 'task1_datast.xlsx'
data = pd.read_excel(path)
data.head()


# In[3]:


data.info()


# In[4]:


data.shape


# In[5]:


print(data.dtypes)


# In[6]:


for col in data.columns:
    print(f"{col}: {data[col].dtype}")


# In[7]:


for col in data.columns:
    print(f"{col}--> {data[col].isnull().sum()}")


# In[8]:


data.describe()


# In[9]:


data.duplicated().sum()


# In[10]:


missing_percent = data.isnull().mean() * 100
high_missing = missing_percent[missing_percent >= 75]
print("Columns with ≥75% missing values and their percentages:")
print(high_missing.sort_values(ascending=False))


# In[11]:


print("Before:", data.shape)
df = data.drop(columns=data.columns[data.isnull().mean() >= 0.75])
print("After:", df.shape)


# In[12]:


for col in df.columns:
    print(col)


# In[13]:


df.columns = df.columns.str.strip()  # removes leading/trailing spaces


# In[14]:


df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


# In[15]:


for col in df.columns:
    print(col)


# In[16]:


law_columns = [
    'national_motorcycle_helmet_law',
    'national_seat-belt_law',
    'national_child_restraints_use_law',
    'national_law_setting_a_speed_limit'
]

for col in law_columns:
    df[col + '_bool'] = df[col].str.strip().str.lower() == 'yes'

df['has_all_key_laws'] = (
    df['national_motorcycle_helmet_law_bool'] &
    df['national_seat-belt_law_bool'] &
    df['national_child_restraints_use_law_bool'] &
    df['national_law_setting_a_speed_limit_bool']
)


# In[17]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[18]:


top10 = df.nlargest(10, 'population')

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(top10['country_name'], top10['population'] / 1_000_000, color='coral')
plt.xticks(rotation=45)
plt.xlabel('Country')
plt.ylabel('Population (in millions)')
plt.title('Top 10 Countries by Population')
# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2,
             height,
             f'{height:.1f}',
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()


# In[19]:


plt.figure(figsize=(12, 6))
sns.histplot(df['2010_who-estimated_rate_per_100_000_population_(update)'], bins=20, kde=True)
plt.title('Distribution of Estimated Road Traffic Deaths per 100,000 Population')
plt.xlabel('Deaths per 100,000')
plt.ylabel('Number of Countries')
plt.show()


# In[20]:


plt.figure(figsize=(16, 8))
plt.bar(df['country_name'], df['population'] / 1_000_000, color='skyblue')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Population (in millions)')
plt.title('Population by Country')
plt.tight_layout()
plt.show()


# In[21]:


top_10_fatalities = df[['country_name', 'reported_fatalities']].sort_values(by='reported_fatalities', ascending=False).head(10)

# Create the horizontal bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(top_10_fatalities['country_name'], top_10_fatalities['reported_fatalities'], color='skyblue')

# Add the numbers on the bars
for bar in bars:
    plt.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():,.0f}',  # Format the number with commas for thousands
             va='center', ha='left', fontsize=10)

plt.xlabel('Reported Fatalities')
plt.ylabel('Country')
plt.title('Top 10 Countries by Reported Fatalities')
plt.gca().invert_yaxis()  # Highest fatalities at the top
plt.show()


# In[25]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='income_group', y='who-estimated_rate_per_100_000_population', palette="Set2")
plt.title('Estimated population rate by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Fatality Rate per 100,000')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[23]:


plt.figure(figsize=(10, 6))
sns.boxplot(x='national_motorcycle_helmet_law', y='2010_who-estimated_rate_per_100_000_population_(update)', data=df, hue = 'national_motorcycle_helmet_law')
plt.title('Helmet Law vs estimated Road Traffic Death Rate')
plt.xlabel('Helmet Law Present')
plt.ylabel('Estimated Road Traffic Deaths per 100,000')
plt.show()


# In[24]:


import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='population', y='who-estimated_rate_per_100_000_population')
plt.xscale('log')
plt.title('Population vs WHO-Estimated Fatality Rate (per 100k)')
plt.xlabel('Population (log scale)')
plt.ylabel('Fatality Rate per 100,000')
plt.grid(True)
plt.tight_layout()
plt.show()

