#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


np.random.seed(42)

# Generate normal data
cluster1 = np.random.normal(loc=[30, 30], scale=5, size=(4998, 2))
cluster2 = np.random.normal(loc=[70, 70], scale=5, size=(4999, 2))
normal_data = np.vstack([cluster1, cluster2])
normal_data = np.clip(normal_data, 0, 100)

# Point anomalies
point_anomalies = np.array([
    [95, 10],
    [5, 95],
    [50, 0]
])
data_point_anomalies = np.vstack([normal_data, point_anomalies])

# Plot point anomalies
plt.figure(figsize=(7, 6))
plt.scatter(normal_data[:, 0], normal_data[:, 1], color='green', label='Normal Data')
plt.scatter(point_anomalies[:, 0], point_anomalies[:, 1], color='red', marker='x', s=100, label='Point Anomalies')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(0, 101, 10))
plt.title("Type 1: Point Anomalies (Low-density Points)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()


# In[3]:


# Regional anomalies
regional_anomalies = np.random.normal(loc=[90, 20], scale=2, size=(3, 2))
regional_anomalies = np.clip(regional_anomalies, 0, 100)
data_regional_anomalies = np.vstack([normal_data, regional_anomalies])

# Plot regional anomalies
plt.figure(figsize=(7, 6))
plt.scatter(normal_data[:, 0], normal_data[:, 1], color='green', label='Normal Data')
plt.scatter(regional_anomalies[:, 0], regional_anomalies[:, 1], color='blue', marker='s', s=100, label='Regional Anomalies')
plt.xlim(0, 100)
plt.ylim(0, 100)
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(0, 101, 10))
plt.title("Type 2: Regional Anomalies (Low-density Region)")
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)
plt.show()

