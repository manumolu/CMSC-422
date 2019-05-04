#!/usr/bin/env python
# coding: utf-8

# In[1]:


import sklearn
import numpy as np
# np.set_printoptions(threshold=np.inf)
import scipy

print('The scikit-learn version is {}.'.format(sklearn.__version__))
print('The numpy version is {}.'.format(np.__version__))
print('The scipy version is {}.'.format(scipy.__version__))


# In[2]:


#RMSE --- check between predicted and target values for test and train data 
#before and after training
import math as mh
def MLP_Rmse(x,y):
#Root Mean Square Error Function
    RMSE = 0
    for i in range(0,len(x)):
        RMSE+= (x[i]-y[i])**2 #----Sum of Square Error
    RMSE = RMSE/len(x)    
    RMSE = mh.sqrt(RMSE)
    return RMSE


# In[3]:


#Get the boston housing dataset
from sklearn.datasets import load_boston
boston = load_boston()


# In[4]:


# print(boston.data.shape)
# type(boston)
#print(boston)
#506X13 bunch with data target feature_names and DESCR as keys (with values)


# In[5]:


# print(boston.data.shape)


# In[6]:


# print(boston.data)


# In[7]:


# print(boston.target)


# In[8]:


# print(boston.feature_names)


# In[9]:


# print(boston.DESCR)


# In[10]:


from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(boston.data, boston.target, random_state = 13)


# In[11]:


# print("x-train") 
# print(np.shape(x_train))


# In[12]:


# print("x-test") 
# print(np.shape(x_test))


# In[13]:


# print("y-train") 
# print(np.shape(y_train))


# In[14]:


# print("y-test") 
# print(np.shape(y_test))


# In[15]:


from sklearn.neural_network import MLPRegressor



# #RMSE 4.9
# model = MLPRegressor(hidden_layer_sizes=(15), learning_rate = 'adaptive',
#                      learning_rate_init = 0.10, max_iter = 4000,momentum = 0.9,random_state = 13)


#RMSE 4.55
model = MLPRegressor(max_iter = 2000,momentum = 0.9,random_state = 13)

# model = MLPRegressor(random_state =13)


# In[16]:


#To get pre-training Weights and Output Values
model.partial_fit(x_train,y_train)


# In[17]:


#Get model parameters
Params = model.get_params(deep = True)
type(Params)
print(Params)


# In[18]:


#Get Inital Parameters to compare pre-training and post-training

InitialWeightsI_H = model.coefs_[0]
# print(InitialWeightsI_H)
print(np.shape(InitialWeightsI_H))

InitialWeightsH_O = model.coefs_[1]
# print(InitialWeightsH_O)
print(np.shape(InitialWeightsH_O))


#Find the bias weights
Initial_BiasI_H = model.intercepts_[0]
Initial_BiasH_O = model.intercepts_[1]


# In[19]:


#Predict on pre-trained network
y_predict_train_pre_train = model.predict(x_train)
y_predict_test_pre_train = model.predict(x_test)


# In[20]:


#Find the RMSE for test and train data on pre-trained network 


# In[21]:


#For train data, pretrain
RMSE_Train_Pre_Train = MLP_Rmse(y_predict_train_pre_train,y_train)


# In[22]:


#For Test Data, pretrain
RMSE_Test_Pre_Train = MLP_Rmse(y_predict_test_pre_train,y_test)


# In[23]:


#Train the neural network
model.fit(x_train,y_train)


# In[24]:


#Predict after training
y_predict_train_post_train = model.predict(x_train)
y_predict_test_post_train = model.predict(x_test)
# print(y_predict)


# In[25]:


#Get the weights matrix of the neural network 

FinalWeightsI_H = model.coefs_[0]
# print(FinalWeightsI_H)
print(np.shape(FinalWeightsI_H))

FinalWeightsH_O = model.coefs_[1]
# print(FinalWeightsH_O)
# print(np.shape(FinalWeightsH_O))

Final_BiasI_H = model.intercepts_[0]
Final_BiasH_O = model.intercepts_[1]


# In[26]:


#Find the RMSE for test and train data post-training


# In[27]:


#for train data, post train
RMSE_Train_Post_Train = MLP_Rmse(y_predict_train_post_train,y_train)


# In[28]:


#for test data, post train
RMSE_Test_Post_Train = MLP_Rmse(y_predict_test_post_train,y_test)


# In[32]:


file = open('/Users/anumolubhargav/Documents/Machine Learning/Homework2/housesBestResults.txt', 'w+')
file.write("This is the best run \n\n\n")
file.write("PARAMETER DATA: \n {} \n ".format(Params)+ "\n" + "\n")
file.write("Random Seed used: 13\n\n")
file.write("Number of Layers:  1\n\n")
file.write("Number of Hidden Nodes:  100\n\n")
file.write("Number of Epochs : 2000 (best run)\n\n")

#Weights and Biases before Training:
file.write("Initial I-->H Weights:\n {} \n".format(InitialWeightsI_H) + "\n")
file.write("Initial I-->H Bias:\n {} \n".format(Initial_BiasI_H) + "\n")

file.write("Initial H-->O Weights:\n {} \n".format(InitialWeightsH_O) + "\n")
file.write("Initial H-->O Bias:\n {} \n".format(Initial_BiasH_O) + "\n")

#Target Values for Test Data and Actual Values for Test Data before training:

file.write("Target Values for Test Data before Training: \n {} \n".format(y_test) + "\n")
file.write("Output Values for Test Data before Training: \n {} \n".format(y_predict_test_pre_train) + "\n")


#RMSE Values before training:
file.write("RMSE Pre-Train Train Data:\n {} \n".format(RMSE_Train_Pre_Train) + "\n")
file.write("RMSE Pre-Train Test Data:\n {} \n".format(RMSE_Test_Pre_Train) + "\n")



#Weights and Biases after training:
file.write("Final I-->H Weights:\n {} \n".format(FinalWeightsI_H) + "\n")
file.write("Final I-->H Bias:\n {} \n".format(Final_BiasI_H) + "\n")

file.write("Final H-->O Weights:\n {} \n".format(FinalWeightsH_O) + "\n")
file.write("Final H-->O Bias:\n {} \n".format(Final_BiasH_O) + "\n")

#Target Values for Test Data and Actual Values for Test Data before training:

file.write("Target Values for Test Data post Training: \n {} \n".format(y_test) + "\n")
file.write("Output Values for Test Data post Training: \n {} \n".format(y_predict_test_post_train) + "\n")

#RMSE Values after training:
file.write("RMSE Post-Train Train Data:\n {} \n".format(RMSE_Train_Post_Train) + "\n")
file.write("RMSE Post-Train Test Data:\n {} \n".format(RMSE_Test_Post_Train) + "\n")


# In[ ]:





# In[ ]:




