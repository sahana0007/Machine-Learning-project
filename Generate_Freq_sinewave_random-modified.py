#!/usr/bin/env python
# coding: utf-8

# In[11]:


import numpy as np
import matplotlib.pyplot as plt
import random

fs = 32000 #sampling rate
Ts = 1/fs #time duration
#t=0.1 
freqList = [250,500, 750 ,1000, 1500, 2000, 3000, 4000, 6000] 
n = np.arange(0,2048,1) #no of samples
print("n = ",n)
noise = np.random.normal(2048) 
phase = 2* np.pi *np.random.random(2048)
for freq in freqList:
    y = np.sin(2* np.pi * freq *n * Ts) + noise + phase
    plt.plot(n,y)
    plt.show()


# In[ ]:




