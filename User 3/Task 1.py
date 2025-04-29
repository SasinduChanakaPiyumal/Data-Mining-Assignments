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


for col in data.columns:
    print(f"{col}: {data[col].dtype}")


# In[6]:


for col in data.columns:
    print(f"{col}--> {data[col].isnull().sum()}")


# In[7]:


data.describe()


# In[8]:


data.duplicated().sum()


# In[9]:


missing_percent = data.isnull().mean() * 100
high_missing = missing_percent[missing_percent >= 75]
print("Columns with ≥75% missing values and their percentages:")
print(high_missing.sort_values(ascending=False))


# In[10]:


print("Before:", data.shape)
df = data.drop(columns=data.columns[data.isnull().mean() >= 0.75])
print("After:", df.shape)


# In[11]:


for col in df.columns:
    print(col)


# In[12]:


df.columns = df.columns.str.strip()  # removes leading/trailing spaces


# In[13]:


df.columns = df.columns.str.strip().str.lower().str.replace(' ', '_')


# In[14]:


for col in df.columns:
    print(col)


# In[15]:


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


# In[16]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[17]:


plt.figure(figsize=(16, 8))
plt.bar(df['country_name'], df['population'] / 1_000_000, color='green')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Population (in millions)')
plt.title('Population by Country')
plt.tight_layout()
plt.show()


# In[18]:


top_10_fatalities = df[['country_name', 'reported_fatalities']].sort_values(by='reported_fatalities', ascending=False).head(10)

# Create the bar chart
plt.figure(figsize=(12, 8))
bars = plt.barh(top_10_fatalities['country_name'], top_10_fatalities['reported_fatalities'], color='red')

# Add the numbers on the bars
for bar in bars:
    plt.text(bar.get_width() + 500, bar.get_y() + bar.get_height() / 2,
             f'{bar.get_width():,.0f}',  # Format the number with commas for thousands
             va='center', ha='left', fontsize=10)

plt.xlabel('Reported Fatalities')
plt.ylabel('Country')
plt.title('Top 10 Countries by Reported Fatalities')
plt.gca().invert_yaxis()  # Invert to display the country with the highest fatalities at the top
plt.show()


# In[19]:


import seaborn as sns
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='population', y='reported_fatalities')
plt.xscale('log')
plt.title('Population vs reported_fatalities')
plt.xlabel('Population (log scale)')
plt.ylabel('reported_fatalities')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[20]:


top10 = df.nlargest(10, 'year_by_fatality_reduction_target')

# Plot
plt.figure(figsize=(12, 6))
bars = plt.bar(top10['country_name'], top10['year_by_fatality_reduction_target'], color='purple')
plt.xticks(rotation=45)
plt.xlabel('Country')
plt.ylabel('Year by fatality reduction target')
plt.title('Top 10 Countries of year by fatality reduction target')
# Add labels on top of bars
for bar in bars:
    height = bar.get_height()
    plt.text(bar.get_x() + bar.get_width() / 2,
             height,
             f'{height:.1f}',
             ha='center', va='bottom', fontsize=10)
plt.tight_layout()
plt.show()


# In[21]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='has_all_key_laws', y= df['who-estimated_rate_per_100_000_population'])
plt.title('Key Law presence vs Road Fatality Rate')
plt.xlabel('The key Laws Exists')
plt.ylabel('Fatality Rate per 100,000')
plt.tight_layout()
plt.show()


# In[22]:


df['national_road_safety_strategy'].value_counts()


# In[23]:


# Step 1: Set a threshold for low-frequency categories
threshold = 2  # You can change this based on your dataset size

# Step 2: Get value counts and identify rare categories
value_counts = df['national_road_safety_strategy'].value_counts()
to_remove = value_counts[value_counts < threshold].index

# Step 3: Remove rows with those rare categories (optional)
df = df[~df['national_road_safety_strategy'].isin(to_remove)]


# In[24]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='national_road_safety_strategy', y='who-estimated_rate_per_100_000_population')
plt.title('Road Safety Strategy Presence vs Fatality Rate')
plt.xlabel('Road Safety Strategy Exists')
plt.ylabel('Estimated Fatality Rate per 100,000')
plt.tight_layout()
plt.show()


# In[25]:


years = ['grssr_participation_2009', 'grssr_participation_2013', 
         'grssr_participation_2015', 'grssr_participation_2018']
participation = df[years].count()

ax = participation.plot(kind='bar', color='orange')
plt.title('Country Participation in Global Road Safety Status Reports')
plt.xlabel('Year')
plt.ylabel('Number of Participating Countries')
plt.xticks(rotation=0)

# Annotate each bar
for bar in ax.patches:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width() / 2,
            height,
            f'{height:.1f}',
            ha='center', va='bottom', fontsize=10)

plt.tight_layout()
plt.show()

