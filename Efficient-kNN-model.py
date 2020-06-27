#!/usr/bin/env python
# coding: utf-8

# In[623]:


import pandas as pd
from sklearn import preprocessing
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
user_col = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J']
data = pd.read_table("abalone.data" , sep = ',', header = None, names = user_col)


# In[624]:


data


# In[109]:


data['A'].unique()


# In[110]:


data['A'].value_counts()


# In[111]:


data.A = pd.Categorical(data.A).codes


# In[112]:


data


# In[113]:


data['A'].value_counts()


# In[719]:


class kNNClassification:
    
    def __init__(self):
        
        np.random.seed(0)
        pass
    
    def normalise_data(self, X):
        
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        
        return X_scaled
    
    def forward_pass(self, X, W):
        
        return np.multiply(X, W)
    
    def error_normal(self, X, W):
        
        #print("normal:", np.sum(np.sqrt((self.forward_pass(X, W) - X) **2)))
        return np.sqrt((self.forward_pass(X, W) - X) **2)
    
    def error_l1_norm(self, X, W, rho1):
        
        #print("rho1:", rho1)
        #print("l1:", np.sum(np.sqrt((self.forward_pass(X, W) - X) **2) + rho1 * np.sqrt(W**2)))
        return np.sqrt((self.forward_pass(X, W) - X) **2) + rho1 * np.sqrt(W**2)
    
    def error_l2_norm(self, X, W, rho2):
        
        #print("rho2:", rho2)
        #print("l2:", np.sum(np.sqrt((self.forward_pass(X, W) - X) **2) + rho2 * abs(W)))
        return np.sqrt((self.forward_pass(X, W) - X) **2) + rho2 * abs(W)
    
    def loss(self, X, W):
        
        return np.sum(self.error_normal( X, W)) / X.shape[0]
    
    def grad_normal(self, X):
        
        return X
    
    def grad_l1_norm(self, X, W, rho1):
        
        if rho1 == 0:
            return X
        else:
            return X + 2*W
    
    def grad_l2_norm(self, X, rho2):
        
        if rho2 == 0:
            return X
        else:
            return X + 1
    
    def fit(self, X, epochs = 150, mini_batch_size = 32, learning_rate = 0.01, 
            method = "Normal", rho1 = 1, rho2 = 1, display_loss = True):
        
        m = X.shape[0]
        X = self.normalise_data(X)
        W = np.random.randn(X.shape[0], X.shape[1])
        Loss  = []
        if method == "Normal":
            #print("Weights:",np.sum(W))
            for i in tqdm(range(epochs)):
                for i in range(0, m, mini_batch_size):
                    self.error_normal(X[i:i+mini_batch_size], W[i:i+mini_batch_size])
                    W[i:i+mini_batch_size] += learning_rate * self.grad_normal(X[i:i+mini_batch_size])
                Loss.append(self.loss(X, W))
                
        elif method == "l1-norm":
            #print("Weights:",np.sum(W))
            for i in tqdm(range(epochs)):
                for i in range(0, m, mini_batch_size):
                    self.error_l1_norm(X[i:i+mini_batch_size], W[i:i+mini_batch_size], rho1)
                    W[i:i+mini_batch_size] -= learning_rate * self.grad_l1_norm(X[i:i+mini_batch_size], W[i:i+mini_batch_size],
                                                                               rho1)
                Loss.append(self.loss(X, W))
            
        elif method == "l2-norm":
            #print("Weights:",np.sum(W))
            for i in tqdm(range(epochs)):
                for i in range(0, m, mini_batch_size):
                    self.error_l2_norm(X[i:i+mini_batch_size], W[i:i+mini_batch_size], rho2)
                    W[i:i+mini_batch_size] += learning_rate * self.grad_l2_norm(X[i:i+mini_batch_size], rho2)
                Loss.append(self.loss(X, W))
                    
            
        if display_loss:
            plt.style.use('ggplot')
            plt.figure(figsize = (10, 8))
            plt.plot(Loss, '-o', markersize = 5)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        return W, min(Loss)


# In[720]:


model = kNNClassification()
Weights_normal, loss_normal = model.fit(X, epochs = 150, mini_batch_size = 32, method = "Normal")
print("Minimum Loss Normal:", loss_normal)


# In[723]:


model = kNNClassification()
Weights_l1_norm, loss_l1 = model.fit(X, epochs = 20, mini_batch_size = 32, method = "l1-norm", rho1 = 1)
print("Minimum L1 Loss:", loss_l1)


# In[722]:


model = kNNClassification()
Weights_l2_norm, loss_l2 = model.fit(X, epochs = 60, mini_batch_size = 32, method = "l2-norm", rho2 = 1)
print("Minimum Loss_l2:", loss_l2)


# In[724]:


Weights_l2_norm


# In[725]:


Weights_l2_norm[1]


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




