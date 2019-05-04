#!/usr/bin/env python
# coding: utf-8

# In[224]:


# modified from Marsland's ML code

import numpy as np

# Perceptron Class Definition
class pcn:

	def __init__(self,inputs,targets,outfid):   # create and initialize network
            self.fid = outfid
            self.trace = False
            if np.ndim(inputs)>1:
                self.nIn = np.shape(inputs)[1]    # num. of input nodes
            else: 
                self.nIn = 1
            if np.ndim(targets)>1:
                self.nOut = np.shape(targets)[1]  # num. of output nodes
            else:
                self.nOut = 1
            self.nData = np.shape(inputs)[0]          # num. of data examples
            # Initialise weights:  nIn+1 x nOut weight matrix (+1 for bias)
                    #                      random wts. between +/-0.05
            self.weights = np.random.rand(self.nIn+1,self.nOut)*0.1-0.05
            ####Different Weight Initialization
            
            self.fid.write("Initial Weights:\n {} \n".format(self.weights) + "\n")

	def pcntrain(self,inputs,targets,eta,nIterations):
            self.nData = np.shape(inputs)[0]          # num. of data examples
		# Add the inputs that match the bias node
            inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)
		# Training
		# change = range(self.nData)  # use iff re-order training data each epoch
            for n in range(nIterations):
                self.activations = self.pcnfwd(inputs) # one epoch's results
                self.weights -= eta*np.dot(np.transpose(inputs),self.activations-targets)

		   # Randomise order of inputs
		   #np.random.shuffle(change)
		   #inputs = inputs[change,:]
		   #targets = targets[change,:]
            
            
            self.fid.write("Final Weights:\n {} \n".format(self.weights) + "\n")
            
            ###########----Manohar Anumolu (UID 115733039)-----########
            self.fid.write("Learning Rate:\n {} \n".format(eta) + "\n" + "\n" )
            self.fid.write("Number of Epochs:\n {} \n".format(nIterations) + "\n" + "\n" )
            
            ##########----------------------------------------#########
		#return self.weights

	def pcnfwd(self,inputs):                            # Run the network (batch mode)
            activations =  np.dot(inputs,self.weights)  # compute weighted node inputs
            return np.where(activations>0,1,0)          # return node outputs 0/1 

        # Confusion Matrix layout:           targets
        #                                      0 1
        #                                      x x  0  actual
        #                                      x x  1

	def confmat(self,inputs,targets,msg):       # generate the confusion matrix
		# Add the inputs that match the bias node
		# inputs = np.concatenate((inputs,-np.ones((self.nData,1))),axis=1)    # BAD 
            inputs = np.concatenate((inputs,-np.ones((inputs.shape[0],1))),axis=1) # FIX
            outputs = np.dot(inputs,self.weights) 
            if self.trace:        # Tracing: show target and actual outputs 
                self.fid.write(msg + ": target actual\n")
                for i in range(self.nData):
                    if outputs[i] > 0:
                        actual = 1
                    else:
                        actual = 0
                    self.fid.write("   {}   {}\n".format(int(targets[i][0]),actual))
                self.fid.write("\n") 
            nClasses = np.shape(targets)[1] 
            if nClasses==1:
                nClasses = 2
                outputs = np.where(outputs>0,1,0)
            else:
			# 1-of-N encoding
                outputs = np.argmax(outputs,1)
                targets = np.argmax(targets,1) 
            cm = np.zeros((nClasses,nClasses))
            for i in range(nClasses):         # for each actual output
                for j in range(nClasses):      # for each target value
                    cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0)) 
                      # rows are actual values, cols are target values 
            self.fid.write(msg + "confusion matrix:\n {} \n".format(cm))
            self.fid.write("Fraction correct: {} \n".format(np.trace(cm)/np.sum(cm)) + "\n")



# In[225]:


import numpy as np
data = np.loadtxt("/Users/anumolubhargav/Documents/Machine Learning/Homework1/radarData.txt",delimiter =",")
# print(data)
np.shape(data)
#Observed 351 Rows and 35 Columns---last column is output data


# In[226]:


#Split into test and train data--take every 5th sample for testing
test = data[::5]
train = np.delete(data, np.s_[::5], axis = 0)

#Split into test and train inputs and outputs

X_train = np.delete(train, -1, 1)
# print(np.shape(X_train))
# print(" X Train : " )
# print(X_train)

Y_train = np.delete(train, np.s_[:-1], 1)
# print(np.shape(Y_train))
# print(" Y Train : " )
# print(Y_train)

X_test = np.delete(test, -1, 1)
# print(np.shape(X_test))
# print(" X Test : " )
# print(X_test)

Y_test = np.delete(test, np.s_[:-1], 1)
# print(np.shape(Y_test))
# print(" Y Test : " )
# print(Y_test)


# In[227]:


##Required in output file

#1.Number of training examples
#2.Number of testing examples
#3.Learning Rate
#4.Number of epochs

#5. Initial Pre-Training Weights
#6. Confusion Matrix - pre-training on training and testing data

#7. Post training Weights
#8. Confusion Matrix - post-training on training and testing data


# In[239]:


#Run the neural network
file = open('/Users/anumolubhargav/Documents/Machine Learning/Homework1/Results.txt', 'w+')

# pre_train = X_train.copy() #for the pre-trained forward pass for training data
# pre_test = X_test.copy()   # for the pre-trained forward pass for testing data


####-NETWORK INITIALIZATION 
#Initialize network to generate pre-trained weights
model = pcn(X_train,Y_train, file)

##----------------------------------------------------------------------------------##
# TRYING FORWARD PASS BEFORE TRAINING ON TESTING AND TRAINING DATA
#Add bias nodes to all inputs (for both pre_test and pre_train)
# pre_train = np.concatenate((pre_train,-np.ones((pre_train.shape[0],1))),axis=1)
# pre_trest = np.concatenate((pre_test,-np.ones((pre_test.shape[0],1))),axis=1)

# #Run the forward pass on pre-trained weights for training data
# pre_trained_training_output = model.pcnfwd(pre_train)

# #Run the forward pass on pre_trained weights for testing data
# pre_trained_testing_output = model.pcnfwd(pre_test)
##---------------------------------------------------------------------------------##

#Find the confusion matrix for pre-trained training data
model.confmat(X_train,Y_train,"Pre-Trained Training Data -- ")

#Find the confusion matrix for pre-trained testing data
model.confmat(X_test,Y_test,"Pre-Trained Testing Data -- ")

#TRAINING THE NEURAL NETWORK
file.write("Training the neural network now....... \n\n\n")
model.pcntrain(X_train, Y_train,0.25,500) #--train the neural network
file.write("Number of Training Examples: {}".format(np.shape(X_train)[0])+ "\n")

#Find the confusion matrix for post-trained training data
model.confmat(X_train,Y_train,"Post_Trained Training Data --")

#Find the confusion matrix for post-trained testing data
model.confmat(X_test,Y_test,"Post-Trained Testing Data -- ")





