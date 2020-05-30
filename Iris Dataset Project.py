#!/usr/bin/env python
# coding: utf-8

# # Basic Analysis of the Iris Data set

# ##### What is our Objective?
# Given the sepal length, sepal width, petal length and petal width, classify the Iris flower into one of the three species — Setosa, Virginica and Versicolor.

# In[18]:


import numpy as np
import pandas as pd 
from pandas import Series, DataFrame

import seaborn as sns
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# In[23]:


data=pd.read_csv("C:/Users/Shivji Tiwari/Desktop/Applied Data Science IBM/Iris dataset/Iris dataset.csv")


# In[60]:


data.head()


# In[14]:


data.tail()


# In[25]:


data.sample(5)


# In[24]:


data.isnull()


# ### Basic Statistical Analysis — Central Tendency and Spread of Data

# ###### find out the mean and median of the different species present in the data.

# In[30]:


data.groupby('variety').agg(['mean', 'median'])


# ###### For all the species, seeing the values of the mean and median of it’s features. It fournd pretty close. This indicates that data is nearly symmetrically distributed with very less presence of outliers. 

# ### Computing the Standard deviation —

# In[29]:


data.groupby('variety').std()


# ###### Standard deviation (or variance) is an indication of how widely the data is spread about the mean.

# #### Box Plot : also known as a box and whisker plot, displays a summary of a large amount of data in five numbers — minimum, lower quartile(25th percentile), median(50th percentile), upper quartile(75th percentile) and maximum data values.
# #Plotting the box-plots using Seaborn library —

# In[42]:


sns.set(style="ticks") 
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.boxplot(x='variety',y='sepal.length',data=data)
plt.subplot(2,2,2)
sns.boxplot(x='variety',y='sepal.width',data=data)
plt.subplot(2,2,3)
sns.boxplot(x='variety',y='petal.length',data=data)
plt.subplot(2,2,4)
sns.boxplot(x='variety',y='petal.width',data=data)
plt.show()


# ###### The isolated points that can be seen in the box-plots above are the outliers in the data. Since these are very few in number, it wouldn't have any significant impact on our analysis.

# ###### A violin plot plays a similar role as a box and whisker plot. It shows the distribution of data across several levels of one (or more) categorical variables(flower species in our case) such that those distributions can be compared. Unlike box plot, in which all of the plot components correspond to actual data points, the violin plot additionally shows the kernel density estimation of the underlying distribution.

# In[58]:


sns.set(style="white")
plt.figure(figsize=(12,10))
plt.subplot(2,2,1)
sns.violinplot(x="variety",y="sepal.length",data=data)
plt.subplot(2,2,2)
sns.violinplot(x="variety",y="sepal.width",data=data)
plt.subplot(2,2,3)
sns.violinplot(x="variety",y="petal.length",data=data)
plt.subplot(2,2,4)
sns.violinplot(x="variety",y="petal.width",data=data)
plt.show()


# ###### Violin plots typically are more informative as compared to the box plots as violin plots also represent the underlying distribution of the data in addition to the statistical summary

# ##### Probability Density Function (PDF) & Cumulative Distribution Function (CDF)
# Uni-variate as the name suggests is one variable analysis. Our ultimate aim is to be able to correctly identify the specie of Iris flower given it’s features — sepal length, sepal width, petal length and petal width. Which among the four features is more useful than other variables in order to distinguish between the species of Iris flower ? To answer this, we will plot the probability density function(PDF) with each feature as a variable on X-axis and it’s histogram and corresponding kernel density plot on Y-axis.
# 

# In[73]:


sns.FacetGrid(data, hue="variety", height=5)    .map(sns.distplot, "sepal.length")    .add_legend();
sns.FacetGrid(data, hue="variety", height=5)    .map(sns.distplot, "sepal.width")    .add_legend();
sns.FacetGrid(data, hue="variety", height=5)    .map(sns.distplot, "petal.length")    .add_legend();
sns.FacetGrid(data, hue="variety", height=5)    .map(sns.distplot, "petal.width")    .add_legend();
plt.show()


# In[101]:


petal_length=float(input("please enter number"))
if petal_length<2.1:
    print("setosa")
elif petal_length>2.1 and petal_length<4.8:
    print("versicolor")
else:
    print("verginica")


# In[ ]:





# In[ ]:




