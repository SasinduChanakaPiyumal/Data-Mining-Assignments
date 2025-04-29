#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


np.random.seed(0)

# Normal data points: Clustered around (50, 50)
normal_data_x = np.random.normal(loc=50, scale=10, size=100)
normal_data_y = np.random.normal(loc=50, scale=10, size=100)

# Anomalies: Data points in low density regions
# Type 1: Anomalies in the low density regions
anomaly_1_x = [10, 90, 10]
anomaly_1_y = [10, 10, 90]

# Type 2: Anomalies in low density regions (further away from the cluster)
anomaly_2_x = [0, 100, 0]
anomaly_2_y = [0, 0, 100]

# Plot 1: Anomalies defined by datapoints in low-density regions
plt.figure(figsize=(8, 6))

# Plot normal data points in green
plt.scatter(normal_data_x, normal_data_y, color='green', label='Normal Data', alpha=0.6)

# Plot anomaly type 1 in red
plt.scatter(anomaly_1_x, anomaly_1_y, color='red', label='Anomaly Type 1', s=100, marker='X')

# Set axis limits
plt.xlim(0, 100)
plt.ylim(0, 100)

# Add grid with steps of 10 by 10
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(0, 101, 10))
plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.7)

# Labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Anomalies: Low-Density Regions (Type 1)')
plt.legend()

# Show plot
plt.show()


# In[3]:


plt.figure(figsize=(8, 6))

# Plot normal data points in green
plt.scatter(normal_data_x, normal_data_y, color='green', label='Normal Data', alpha=0.6)

# Plot anomaly type 2 in blue
plt.scatter(anomaly_2_x, anomaly_2_y, color='orange', label='Anomaly Type 2', s=100, marker='X')

# Set axis limits
plt.xlim(0, 100)
plt.ylim(0, 100)

# Add grid with steps of 10 by 10
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(0, 101, 10))
plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.7)

# Labels and title
plt.xlabel('X-axis')
plt.ylabel('Y-axis')
plt.title('Anomalies: Regions of Low Density (Type 2)')
plt.legend()

# Show plot
plt.show()


# In[ ]:




