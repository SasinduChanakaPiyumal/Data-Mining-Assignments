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


plt.figure(figsize=(10, 6))
sns.boxplot(x='national_motorcycle_helmet_law', y='2010_who-estimated_rate_per_100_000_population_(update)', data=df, hue = 'national_motorcycle_helmet_law')
plt.title('Helmet Law vs estimated Road Traffic Death Rate')
plt.xlabel('Helmet Law Present')
plt.ylabel('Estimated Road Traffic Deaths per 100,000')
plt.show()


# In[19]:


plt.figure(figsize=(16, 8))
plt.bar(df['country_name'], df['population'] / 1_000_000, color='skyblue')
plt.xticks(rotation=90)
plt.xlabel('Country')
plt.ylabel('Population (in millions)')
plt.title('Population by Country')
plt.tight_layout()
plt.show()


# In[20]:


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


# In[21]:


plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='income_group', y='who-estimated_rate_per_100_000_population', palette="Paired")
plt.title('Estimated population rate by Income Group')
plt.xlabel('Income Group')
plt.ylabel('Fatality Rate per 100,000')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()


# In[22]:


columns = [
    'reported_fatalities_user_distribution__(%_pedestrian)',
    'reported_fatalities_user_distribution__(%_cyclist)',
    'reported_fatalities_user_distribution__(%_powered_2/_wheelers)',
    'reported_fatalities_user_distribution__(%_powered_light_vehicles)',
    'reported_fatalities_user_distribution__(%_other)'
]

df_user_dist = df[columns].mean().sort_values()

df_user_dist.plot(kind='barh', figsize=(10, 6), color='orange')
plt.title('Average Fatality Distribution by Road User Type')
plt.xlabel('Average Percentage')
plt.tight_layout()
plt.show()


# In[23]:


df['national_good_samaritan_law'].value_counts().plot(kind='bar', color='slateblue')
plt.title('Presence of Good Samaritan Law by Country')
plt.xlabel('Good Samaritan Law Exists')
plt.ylabel('Number of Countries')
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()


# In[24]:


df['bac_limit_–_general_population'] = pd.to_numeric(df['bac_limit_–_general_population'], errors='coerce')
df['who-estimated_rate_per_100_000_population'] = pd.to_numeric(df['who-estimated_rate_per_100_000_population'], errors='coerce')

# Now plot the scatter plot
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df, x='bac_limit_–_general_population', y='who-estimated_rate_per_100_000_population', hue='income_group', palette='coolwarm')
plt.title('BAC Limit vs Road Fatality Rate')
plt.xlabel('Blood Alcohol Concentration Limit (%)')
plt.ylabel('WHO Estimated Rate per 100,000 Population')
plt.tight_layout()
plt.show()


# In[25]:


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


# In[26]:


import numpy as np
sorted_indices = np.argsort(df['population'] / 1_000_000)[::-1]  # Sorting in descending order
top_10_countries = np.array(df['country_name'])[sorted_indices][:10]
top_10_population = np.array(df['population'])[sorted_indices][:10]
top_10_fatality_rate = np.array(df['reported_fatalities'])[sorted_indices][:10]

# Create figure and axis objects
fig, ax1 = plt.subplots(figsize=(10, 6))

# Plot population on the first y-axis
ax1.plot(top_10_countries, top_10_population, color='blue', marker='o', label='Population', linestyle='-', linewidth=2)
ax1.set_xlabel('Countries')
ax1.set_ylabel('Population (per 100,000)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

# Create a second y-axis that shares the same x-axis
ax2 = ax1.twinx()

# Plot fatality rate on the second y-axis
ax2.plot(top_10_countries, top_10_fatality_rate, color='red', marker='s', label='Fatality Rate', linestyle='--', linewidth=2)
ax2.set_ylabel('Reported fatalities', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# Title and layout
plt.title('Top 10 Countries by Population and Fatalities')
fig.tight_layout()  # Adjust layout to prevent clipping of labels

# Show plot
plt.show()


# In[27]:


df['maximum_urban_speed_limit'] = pd.to_numeric(df['maximum_urban_speed_limit'], errors='coerce')
df['maximum_rural_speed_limit'] = pd.to_numeric(df['maximum_rural_speed_limit'], errors='coerce')

# Drop rows where any of the relevant speed limit columns have NaN values
df_cleaned = df.dropna(subset=['maximum_urban_speed_limit', 'maximum_rural_speed_limit'])

# Sort the dataframe by the maximum urban speed limit (or any other criteria) and select top 10 countries
df_sorted_urban = df_cleaned[['country_name', 'maximum_urban_speed_limit']].sort_values(by='maximum_urban_speed_limit', ascending=False).head(10)
#df_sorted_rural = df_cleaned[['country_name', 'maximum_rural_speed_limit']].sort_values(by='maximum_rural_speed_limit', ascending=False).head(10)

# Create a figure for plotting
plt.figure(figsize=(12, 6))

# Plot urban speed limits for the top 10 countries
sns.barplot(x='country_name', y='maximum_urban_speed_limit', data=df_sorted_urban, color='blue', alpha=0.6, label='Urban Speed Limit')

# Plot rural speed limits for the top 10 countries
#sns.barplot(x='country_name', y='maximum_rural_speed_limit', data=df_sorted_rural, color='orange', alpha=0.6, label='Rural Speed Limit')

# Optionally, if you have motorway speed limits, you can plot them as well
# sns.barplot(x='country_name', y='maximum_motorway_speed_limit', data=df_sorted_urban, color='green', alpha=0.6, label='Motorway Speed Limit')

# Add labels, title, and legend
plt.xlabel('Country')
plt.ylabel('Speed Limit (km/h)')
plt.title('Top 10 Countries by Speed Limits (Urban vs Rural)')
plt.xticks(rotation=45)  # Rotate x-axis labels for better readability
plt.legend(title="Speed Limits")

# Show plot
plt.tight_layout()
plt.show()


# In[28]:


df['maximum_urban_speed_limit'] = pd.to_numeric(df['maximum_urban_speed_limit'], errors='coerce')
df['maximum_rural_speed_limit'] = pd.to_numeric(df['maximum_rural_speed_limit'], errors='coerce')

# Drop rows where any of the relevant speed limit columns have NaN values
df_cleaned = df.dropna(subset=['maximum_urban_speed_limit', 'maximum_rural_speed_limit'])

# Sort the dataframe by the maximum urban speed limit (or any other criteria) and select top 10 countries
df_sorted_urban = df_cleaned[['country_name', 'maximum_urban_speed_limit']].sort_values(by='maximum_urban_speed_limit', ascending=False).head(10)
df_sorted_rural = df_cleaned[['country_name', 'maximum_rural_speed_limit']].sort_values(by='maximum_rural_speed_limit', ascending=False).head(10)

# Create subplots
fig, axs = plt.subplots(1, 2, figsize=(15, 6))

# Plot urban speed limits for the top 10 countries (subplot 1)
sns.barplot(x='country_name', y='maximum_urban_speed_limit', data=df_sorted_urban, color='blue', alpha=0.6, ax=axs[0])
axs[0].set_title('Top 10 Urban Speed Limits by Countries')
axs[0].set_xlabel('Country')
axs[0].set_ylabel('Speed Limit (km/h)')
axs[0].tick_params(axis='x', rotation=45)

# Plot rural speed limits for the top 10 countries (subplot 2)
sns.barplot(x='country_name', y='maximum_rural_speed_limit', data=df_sorted_rural, color='orange', alpha=0.6, ax=axs[1])
axs[1].set_title('Top 10 Rural Speed Limits by countries')
axs[1].set_xlabel('Country')
axs[1].set_ylabel('Speed Limit (km/h)')
axs[1].tick_params(axis='x', rotation=45)

# Adjust layout to avoid overlapping text
plt.tight_layout()

# Show plot
plt.show()

