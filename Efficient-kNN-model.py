#!/usr/bin/env python
# coding: utf-8

# In[364]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn import preprocessing
from scipy.sparse import csgraph
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.tree import DecisionTreeClassifier

from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

user_col = ['A', 'B', 'C', 'D', 'F', 'G', 'H', 'I', 'J']
data = pd.read_table("abalone.data" , sep = ',', header = None, names = user_col)


# In[288]:


data


# In[289]:


data['A'].unique()


# In[290]:


data['A'].value_counts()


# In[291]:


data.A = pd.Categorical(data.A).codes


# In[293]:


data_copy = data


# In[294]:


data_copy


# In[23]:


data['A'].value_counts()


# In[24]:


data = data.to_numpy()


# In[25]:


data


# In[103]:


class EffficientkNNclassification:
    
    def __init__(self):
        
        np.random.seed(0)
        pass
        
    def normalise_data(self, X):
        
        min_max_scaler = preprocessing.MinMaxScaler()
        X_scaled = min_max_scaler.fit_transform(X)
        
        return X_scaled
    
    def forward_pass(self, X, W, rho1, rho2):
        
        a = np.sqrt((np.matmul(X, W) - X) **2)
        b = np.sum(np.absolute(W), axis = 1)
        L = csgraph.laplacian(X[:, 1]*X[:, 1][:, np.newaxis], normed=False)
        c = np.matmul(np.matmul(np.matmul(np.matmul(np.transpose(W), np.transpose(X)), L), X), W) 
        
        return a + (rho1 * b) + (rho2 * np.trace(c))
    
    def loss_fn(self, X, W):
        
        return np.sum(np.sqrt(np.matmul(X, W) ** 2)) / X.shape[1]
    
    def update_weights(self, W):
        
        return np.sqrt(W**2) / (2 * np.sqrt(np.absolute(W)))
    
    
    def fit(self, X, epochs = 1, rho1 = 1, rho2 = 1, display_loss = True):
        
        X = self.normalise_data(X)
        X = np.transpose(X)
        W = np.random.randn(X.shape[1], X.shape[1])
        loss = []
        
        for i in tqdm(range(epochs)):
            
            self.forward_pass(X, W, rho1, rho2)
            loss.append(self.loss_fn(X, W))
            W = np.absolute(W) - 0.1 * self.update_weights(W)
            
        if display_loss:
            plt.style.use('ggplot')
            plt.figure(figsize = (5,5))
            plt.plot(loss, '-o', markersize = 5)
            plt.xlabel('Epoch')
            plt.ylabel('Loss')
            plt.show()
        #print(W)
        #print(W.shape)
        return W
        


# In[104]:


model = EffficientkNNclassification()


# In[108]:


get_ipython().run_cell_magic('time', '', 'Weights = model.fit(data, epochs = 50, rho1 = 1e-4, rho2 = 1e-5)')


# In[109]:


Weights.shape


# In[110]:


Weights


# In[117]:


result = np.where(Weights < 0, 0, Weights)


# In[118]:


result


# In[126]:


result[0]


# In[252]:


i = np.random.randint(1, 4177)
print(i)
print(result[i])
np.count_nonzero(result[i])


# In[253]:


training_label = []
for i in range(len(result)):
    training_label.append(np.count_nonzero(result[i]) - 1)
    


# In[266]:


result_dataframe = pd.DataFrame(training_label, columns = ["Labels"])


# In[271]:


result_dataframe.to_csv('result_csv.csv')


# In[295]:


final_data = pd.concat([data_copy, result_dataframe], axis = 1)


# In[296]:


final_data


# In[309]:


data_numpy = np.array(final_data)


# In[310]:


data_numpy


# In[323]:


labels = data_numpy[:, 9]
labels.shape


# In[322]:


dataset = data_numpy[:, :9]
dataset.shape


# In[383]:


X_train, X_test, Y_train, Y_test = train_test_split(dataset, labels, test_size = 0.1)


# In[384]:


print("X_train", X_train.shape)
print("X_test", X_test.shape)
print("Y_train", Y_train.shape)
print("Y_test", Y_test.shape)


# In[393]:


tree_clf = DecisionTreeClassifier(max_depth = 4)
tree_clf.fit(X_train, Y_train)


# In[394]:


i = np.random.randint(0, 418)
print("index:", i)
print("Dataset:",X_test[i])
print("Predicted:",tree_clf.predict([X_test[i]]))
print("Actual:", Y_test[i])


# In[395]:


Y_pred = tree_clf.predict(X_test)
print("Accuracy:", metrics.accuracy_score(Y_pred, Y_test))


# In[396]:


for i in range(1, 20):
    tree_clf = DecisionTreeClassifier(max_depth = i)
    tree_clf.fit(X_train, Y_train)
    Y_pred = tree_clf.predict(X_test)
    print("Accuracy at depth",i,": ", round(metrics.accuracy_score(Y_pred, Y_test), 4))


# In[506]:


X_dup_test = []
for i in range(len(X_test)):
    X_dup_test.append(X_test[i][1:5])
X_dup_test = np.array( X_dup_test)


# In[507]:


X_dup_train = []
for i in range(len(X_train)):
    X_dup_train.append(X_train[i][1:5])
X_dup_train= np.array(X_dup_train)


# In[508]:


X_dup_train


# In[509]:


tree_clf = DecisionTreeClassifier(max_depth = 3)
tree_clf.fit(X_dup_train, Y_train)


# In[510]:


i = np.random.randint(0, 418)
print("index:", i)
print("Dataset:",X_test[i])
print("Predicted:",tree_clf.predict([X_dup_test[i]]))
print("Actual:", Y_test[i])


# In[511]:


Y_pred = tree_clf.predict(X_dup_test)
print("Accuracy:", metrics.accuracy_score(Y_pred, Y_test))


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




