#!/usr/bin/env python
# coding: utf-8

# In[106]:



# Code from Chapter 4 of Machine Learning: An Algorithmic Perspective (2nd Edition)
# by Stephen Marsland (http://stephenmonika.net)

# You are free to use, change, or redistribute the code in any way you wish for
# non-commercial purposes, but please maintain the name of the original author.
# This code comes with no warranty of any kind.

# Stephen Marsland, 2008, 2014

import numpy as np

class mlp:
    """ A Multi-Layer Perceptron"""
    
#     def __init__(self,inputs,targets,nhidden,beta=1,momentum=0.9,outtype='logistic'):
    def __init__(self,inputs,targets,outfid,nhidden,beta=1,momentum=0.9,outtype='logistic'):
        
        self.fid = outfid ### edit made to write to new file "MLP_Results.txt"    <-------------#
        
        """ Constructor """
        # Set up network size
        
        self.nin = np.shape(inputs)[1]
        self.nout = np.shape(targets)[1]
        self.ndata = np.shape(inputs)[0]
        self.nhidden = nhidden

        self.beta = beta
        self.momentum = momentum
        self.outtype = outtype
    
        # Initialise network
        self.weights1 = (np.random.rand(self.nin+1,self.nhidden)-0.5)*2/np.sqrt(self.nin)
        self.weights2 = (np.random.rand(self.nhidden+1,self.nout)-0.5)*2/np.sqrt(self.nhidden)
        
        #######---------MANOHAR ANUMOLU UID 115733039-------#######
        self.fid.write("Initial I-->H Weights:\n {} \n".format(self.weights1) + "\n")
        self.fid.write("Initial H-->O Weights:\n {} \n".format(self.weights2) + "\n")
        
        #########-------------------------------------------########

    def earlystopping(self,inputs,targets,valid,validtargets,eta,niterations=100):
        
        valid = np.concatenate((valid,-np.ones((np.shape(valid)[0],1))),axis=1)
        
        old_val_error1 = 100002
        old_val_error2 = 100001
        new_val_error = 100000
        
        count = 0
        while (((old_val_error1 - new_val_error) > 0.001) or ((old_val_error2 - old_val_error1)>0.001)):
            count+=1
            print(count)
            self.mlptrain(inputs,targets,eta,niterations)
            old_val_error2 = old_val_error1
            old_val_error1 = new_val_error
            validout = self.mlpfwd(valid)
            new_val_error = 0.5*np.sum((validtargets-validout)**2)
            
        print("Stopped", new_val_error,old_val_error1, old_val_error2)
        return new_val_error
    	
    def mlptrain(self,inputs,targets,eta,niterations):
        """ Train the thing """    
        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((self.ndata,1))),axis=1)
        change = range(self.ndata)
    
        updatew1 = np.zeros((np.shape(self.weights1)))
        updatew2 = np.zeros((np.shape(self.weights2)))
            
        for n in range(niterations):
    
            self.outputs = self.mlpfwd(inputs)

            error = 0.5*np.sum((self.outputs-targets)**2)
            if (np.mod(n,100)==0):
                print("Iteration: ",n, " Error: ",error)    

            # Different types of output neurons
            if self.outtype == 'linear':
            	deltao = (self.outputs-targets)/self.ndata
            elif self.outtype == 'logistic':
            	deltao = self.beta*(self.outputs-targets)*self.outputs*(1.0-self.outputs)
            elif self.outtype == 'softmax':
                deltao = (self.outputs-targets)*(self.outputs*(-self.outputs)+self.outputs)/self.ndata 
            else:
            	print("error")
            
            deltah = self.hidden*self.beta*(1.0-self.hidden)*(np.dot(deltao,np.transpose(self.weights2)))
                      
            updatew1 = eta*(np.dot(np.transpose(inputs),deltah[:,:-1])) + self.momentum*updatew1
            updatew2 = eta*(np.dot(np.transpose(self.hidden),deltao)) + self.momentum*updatew2
            self.weights1 -= updatew1
            self.weights2 -= updatew2
         
        ###########----Manohar Anumolu (UID 115733039)-----########

        self.fid.write("Learning Rate:\n {} \n".format(eta) + "\n" + "\n" )
        self.fid.write("Number of Epochs:\n {} \n".format(niterations) + "\n" + "\n" )
        
        self.fid.write("Final I---->H Weights:\n {} \n".format(self.weights1) + "\n" + "\n")
        self.fid.write("Final H---->O Weights:\n {} \n".format(self.weights2) + "\n" + "\n")

        ##########----------------------------------------######### 
        
            # Randomise order of inputs (not necessary for matrix-based calculation)
            #np.random.shuffle(change)
            #inputs = inputs[change,:]
            #targets = targets[change,:]
            
    def mlpfwd(self,inputs):
        """ Run the network forward """

        self.hidden = np.dot(inputs,self.weights1);
        self.hidden = 1.0/(1.0+np.exp(-self.beta*self.hidden))
        self.hidden = np.concatenate((self.hidden,-np.ones((np.shape(inputs)[0],1))),axis=1)

        outputs = np.dot(self.hidden,self.weights2);

        # Different types of output neurons
        if self.outtype == 'linear':
        	return outputs
        elif self.outtype == 'logistic':
            return 1.0/(1.0+np.exp(-self.beta*outputs))
        elif self.outtype == 'softmax':
            normalisers = np.sum(np.exp(outputs),axis=1)*np.ones((1,np.shape(outputs)[0]))
            return np.transpose(np.transpose(np.exp(outputs))/normalisers)
        else:
            print("error")

    def confmat(self,inputs,targets,msg):
        """Confusion matrix"""

        # Add the inputs that match the bias node
        inputs = np.concatenate((inputs,-np.ones((np.shape(inputs)[0],1))),axis=1)
        outputs = self.mlpfwd(inputs)
        
        nclasses = np.shape(targets)[1]

        if nclasses==1:
            nclasses = 2
            outputs = np.where(outputs>0.5,1,0)
        else:
            # 1-of-N encoding
            outputs = np.argmax(outputs,1)
            targets = np.argmax(targets,1)

        cm = np.zeros((nclasses,nclasses))
        for i in range(nclasses):
            for j in range(nclasses):
                cm[i,j] = np.sum(np.where(outputs==i,1,0)*np.where(targets==j,1,0))
         
        
        #####-------------------Manohar Anumolu UID 115733039----------#######
        self.fid.write(msg + "confusion matrix:\n {} \n".format(cm))
        self.fid.write("Fraction correct: {} \n".format(np.trace(cm)/np.sum(cm)) + "\n")
        ######----------------------------------------------------------#######
        
#         print (msg + "Confusion matrix is:")
#         print (cm)
#         print ("Percentage Correct: ",np.trace(cm)/np.sum(cm)*100)
        


# In[107]:


#Using same snippets as shown in previous homework  - Manohar Anumolu UID 115733039
import numpy as np
data = np.loadtxt("/Users/anumolubhargav/Documents/Machine Learning/Homework2/radarData.txt",delimiter =",")
# print(data)
np.shape(data)
#Observed 351 Rows and 35 Columns---last column is output data


# In[108]:


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


# In[109]:


##Required in output file

#1.Number of training examples
#2.Number of testing examples
#3.Learning Rate
#4.Number of epochs
#5.Number of Hidden Nodes

#5. Initial Pre-Training Weights
#6. Confusion Matrix - pre-training on training and testing data

#7. Post training Weights
#8. Confusion Matrix - post-training on training and testing data


# In[154]:


#Run the neural network
file = open('/Users/anumolubhargav/Documents/Machine Learning/Homework2/MLP_Results.txt', 'w+')

####-NETWORK INITIALIZATION 
#Initialize network to generate pre-trained weights
model = mlp(X_train,Y_train,file,23,beta=1,momentum=0.9,outtype='linear')
file.write("Number of Hidden Nodes: 23 \n\n\n")


##CONFUSION MATRICES -- PRE-TRAINING
#Find the confusion matrix for pre-trained training data
model.confmat(X_train,Y_train,"Pre-Trained Training Data -- ")
#Find the confusion matrix for pre-trained testing data
model.confmat(X_test,Y_test,"Pre-Trained Testing Data -- ")

#TRAINING THE NEURAL NETWORK
file.write("\n\n\n Training the neural network now....... \n\n\n")
file.write("Number of Training Examples: \n {} \n ".format(np.shape(X_train)[0])+ "\n" + "\n")
model.mlptrain(X_train, Y_train,0.25,3700) #--train the neural network

####CONFUSION MATRICES -- POST-TRAINING
#Find the confusion matrix for pre-trained training data
model.confmat(X_train,Y_train,"Post-Trained Training Data -- ")
#Find the confusion matrix for pre-trained testing data
model.confmat(X_test,Y_test,"Post-Trained Testing Data -- ")

