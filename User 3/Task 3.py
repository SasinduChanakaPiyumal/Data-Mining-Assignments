#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# In[2]:


normal_data1 = np.random.multivariate_normal([50, 50], [[70, 0], [0, 70]], 97)
anomalies1 = np.array([[10, 9.9], [99, 98.5], [8, 78]])  # sparse area points
data1 = np.vstack((normal_data1, anomalies1))
labels1 = np.array([0]*97 + [1]*3)

# (2) Anomalies as regions of low density (hole in center)
theta = np.random.uniform(0, 2*np.pi, 97)
radius = np.random.uniform(40, 50, 97)
x_ring = 50 + radius * np.cos(theta)
y_ring = 50 + radius * np.sin(theta)
normal_data2 = np.vstack((x_ring, y_ring)).T
anomalies2 = np.array([[48, 52], [36, 40], [50, 52]])  # points in the center (hole)
data2 = np.vstack((normal_data2, anomalies2))
labels2 = np.array([0]*97 + [1]*3)

# Plot 1: Anomalies as points in low density regions
plt.figure(figsize=(6, 6))
# Set axis limits
plt.xlim(0, 100)
plt.ylim(0, 100)

# Add grid with steps of 10 by 10
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(0, 101, 10))
plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.7)


plt.scatter(data1[labels1 == 0, 0], data1[labels1 == 0, 1], c='green', label='Normal')
plt.scatter(data1[labels1 == 1, 0], data1[labels1 == 1, 1], c='red', label='Anomaly')
plt.title("(1) Anomalies = Points in Low Density Regions")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()


# In[3]:


plt.figure(figsize=(6, 6))

plt.xlim(0, 100)
plt.ylim(0, 100)

# Add grid with steps of 10 by 10
plt.xticks(np.arange(0, 101, 10))
plt.yticks(np.arange(0, 101, 10))
plt.grid(True, which='both', linestyle='--', color='gray', alpha=0.7)

plt.scatter(data2[labels2 == 0, 0], data2[labels2 == 0, 1], c='green', label='Normal')
plt.scatter(data2[labels2 == 1, 0], data2[labels2 == 1, 1], c='red', label='Anomaly')
plt.title("(2) Anomalies = Regions of Low Density")
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.show()

