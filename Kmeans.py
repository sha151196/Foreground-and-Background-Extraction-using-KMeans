#!/usr/bin/env python
# coding: utf-8

# In[52]:


from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import numpy as np
import copy
import matplotlib.image as mpimg


# In[53]:


img = mpimg.imread("sample.jpeg")
shape = img.shape


# In[54]:


#Converting matrix to vector
img = img.reshape((-1,3))


# In[55]:


# convert to np.float32
img1 = np.float32(img)


# In[56]:


# KMeans Clustering
kmeans = KMeans(n_clusters = 2)
kmeans.fit(img1)
y_kmeans = kmeans.predict(img)


# In[57]:


#Plotting the clusters and cluster centres
plt.scatter(img1[:, 0], img1[:, 1], c=y_kmeans, s=50, cmap='viridis')


# In[58]:


centers = kmeans.cluster_centers_
plt.scatter(centers[:, 0], centers[:, 1], c='black', s=200);


# In[59]:


#labels of each datapoint
labels = kmeans.labels_
labels.size


# In[60]:


#Creating deepcopy of original images
img_1 = copy.deepcopy(img)
img_2 = copy.deepcopy(img)


# In[61]:


#Extracting the foreground and background
for i in range(labels.size):
    if (labels[i] == 0):
        img_1[i] = img[i]    
    else:
        img_1[i] = [255,255,255]
        
for i in range(labels.size):
    if (labels[i] == 1):
        img_2[i] = img[i] 
    else:
        img_2[i] = [255,255,255]  


# In[62]:


img_1 = img_1.reshape(shape)
img_2 = img_2.reshape(shape)


# In[63]:


plt.imshow(img_1)


# In[64]:


plt.imshow(img_2)


# In[65]:


mpimg.imsave('img_1.png', img_1)
mpimg.imsave('img_2.png', img_2)


# In[ ]:




