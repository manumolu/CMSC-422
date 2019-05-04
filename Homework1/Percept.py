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


