#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd


# In[2]:


import seaborn as sns
import matplotlib.pyplot as plt


# In[3]:


path = 'task1_datast.xlsx'
data = pd.read_excel(path)
data.head()


# In[4]:


data.info()


# In[5]:


data.shape


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


plt.figure(figsize=(12, 6))
sns.histplot(df['population'], bins=20, kde=True)
plt.title('Distribution of population')
plt.xlabel('population')
plt.ylabel('Number of Countries')
plt.show()


# In[18]:


top_10_fatalities = df[['country_name', 'reported_fatalities']].sort_values(by='reported_fatalities', ascending=False).head(10)

# Create the vertical bar chart
plt.figure(figsize=(12, 8))
bars = plt.bar(top_10_fatalities['country_name'], top_10_fatalities['reported_fatalities'], color='green')

# Add the numbers on top of the bars
for bar in bars:
    plt.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1000,
             f'{bar.get_height():,.0f}',  # Format with commas for thousands
             ha='center', va='bottom', fontsize=10)

plt.ylabel('Reported Fatalities')
plt.xlabel('Country')
plt.title('Top 10 Countries by Reported Fatalities')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()


# In[19]:


columns = [
    'reported_fatalities_user_distribution__(%_pedestrian)',
    'reported_fatalities_user_distribution__(%_cyclist)',
    'reported_fatalities_user_distribution__(%_powered_2/_wheelers)',
    'reported_fatalities_user_distribution__(%_powered_light_vehicles)',
    'reported_fatalities_user_distribution__(%_other)'
]

df_user_dist = df[columns].mean().sort_values()

df_user_dist.plot(kind='barh', figsize=(10, 6), color='skyblue')
plt.title('Average Fatality Distribution by Road User Type')
plt.xlabel('Average Percentage')
plt.tight_layout()
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


plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='total_registered_vehicles_rate_per_100_000_pop', y='who-estimated_rate_per_100_000_population', hue='income_group')
plt.title('Vehicles per 100,000 Population vs Fatality Rate')
plt.xlabel('Registered Vehicles per 100,000')
plt.ylabel('Fatality Rate per 100,000')
plt.grid(True)
plt.tight_layout()
plt.show()


# In[22]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='has_all_key_laws', y= df['who-estimated_rate_per_100_000_population'],palette="Set3")
plt.title('Key Law presence vs Road Fatality Rate')
plt.xlabel('The key Laws Exists')
plt.ylabel('Fatality Rate per 100,000')
plt.tight_layout()
plt.show()


# In[23]:


df['national_road_safety_strategy'].value_counts()


# In[24]:


# Step 1: Set a threshold for low-frequency categories
threshold = 2  # You can change this based on your dataset size

# Step 2: Get value counts and identify rare categories
value_counts = df['national_road_safety_strategy'].value_counts()
to_remove = value_counts[value_counts < threshold].index

# Step 3: Remove rows with those rare categories (optional)
df = df[~df['national_road_safety_strategy'].isin(to_remove)]


# In[25]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='national_road_safety_strategy', y='who-estimated_rate_per_100_000_population')
plt.title('Road Safety Strategy Presence vs Fatality Rate')
plt.xlabel('Road Safety Strategy Exists')
plt.ylabel('Estimated Fatality Rate per 100,000')
plt.tight_layout()
plt.show()


# In[26]:


plt.figure(figsize=(10, 6))
sns.boxplot(data=df, x='national_law_on_drink-driving', y='who-estimated_rate_per_100_000_population', palette='pastel')
plt.title('Seatbelt Law vs Seatbelt Wearing Rate (Drivers)')
plt.xlabel('National Seatbelt Law Exists')
plt.ylabel('Seatbelt Wearing Rate (%)')
plt.tight_layout()
plt.show()

