#!/usr/bin/env python
# coding: utf-8

# In[1]:


# ID3-like decision tree induction
# Modified from Chap. 12, Machine Learning, Stephen Marsland, 2015

import numpy as np

class dtree:                       # decision tree class
	def __init__(self,outfid):      # constructor
		""" constructor for DT class """
		self.fid = outfid            # output file
		self.attr_nodes = 0
		self.leaf_nodes = 0

	def read_data(self,filename):   # reads attrs, data in file filename (string)
		fid = open(filename,"r")  
		data = []   # list of input strings, where each sublist is from one line
		d = []      # list of input data
		for line in fid.readlines():
			d.append(line.strip())       # remove leading/trailing white space
		for d1 in d:
			data.append(d1.split(","))   # splits d1 using ',' as delimiter
		fid.close() 
		self.featureNames = data[0]                # list of atr names (1st line fid)
		self.featureNames = self.featureNames[:-1] # remove output class name
		data = data[1:]                 # list of input examples (delete data header)
		self.classes = []               
		for d in range(len(data)):
			self.classes.append(data[d][-1])
			data[d] = data[d][:-1] 
		return data,self.classes,self.featureNames

	def classify(self,tree,datapoint):      # use tree to classify pattern datapoint
		if type(tree) == type("string"):   # have reached a leaf
			return tree                # return the classification
		else:
			a = tree.keys()[0]         # first node in tree
			for i in range(len(self.featureNames)):   # for each input feature i
				if self.featureNames[i]==a:       
					break
			try:
				t = tree[a][datapoint[i]]
				return self.classify(t,datapoint)
			except:
				return None

	def classifyAll(self,tree,data):        # use tree to classify all examples in data
		results = []
		for i in range(len(data)):         # for each example in data
			results.append(self.classify(tree,data[i]))  # run i through the tree
		return results

	def make_tree(self,data,classes,featureNames,maxlevel=-1,level=0,forest=0):
		# main function; recursively constructs the tree
		nData = len(data)          # number of examples
		nFeatures = len(data[0])   # number of input features or attributes
		try: 
			self.featureNames                  # if no feature names
		except:
			self.featureNames = featureNames   # use those given as input
		newClasses = []            # list of unique possible output classes
		for aclass in classes:     # newClasses = unique class names; e.g., ['y', 'n']?
			if newClasses.count(aclass)==0:    # if aclass is not in newClasses
				newClasses.append(aclass)  # then add aclass to newClasses 
		# compute the default class and total entropy of data
		frequency = np.zeros(len(newClasses)) 
		totalEntropy = 0           # entropy of data
		totalGini = 0              # Gini value of data
		index = 0
		for aclass in newClasses:  # for each unique output class
			frequency[index] = classes.count(aclass)  # num times aclass in targets
			totalEntropy += self.calc_entropy(float(frequency[index])/nData)
			totalGini += (float(frequency[index])/nData)**2 
			index += 1 
		totalGini = 1 - totalGini
		default = newClasses[np.argmax(frequency)]  # FIX2: replaces next line
			#default = classes[np.argmax(frequency)]    # BAD? ******
		if nData==0 or nFeatures == 0 or (maxlevel>=0 and level>maxlevel):
			return default       # have reached an empty branch
		elif frequency[np.argmax(frequency)] == nData:   # FIX3: if all data in one class
			return newClasses[np.argmax(frequency)]  # FIX3: return that class
		#elif classes.count(classes[0]) == nData:        # BAD? ****** 
		#	return classes[0]                        # BAD? ******
		else:                        # else choose which input attribute is best	
			gain = np.zeros(nFeatures)     # entropy gains for features
			ggain = np.zeros(nFeatures)    # Gini gains for features
			featureSet = range(nFeatures)  # indices of features (input atrs)
			if forest != 0:
				np.random.shuffle(featureSet)
				featureSet = featureSet[0:forest]
			for feature in featureSet:     # for each input feature
				g,gg = self.calc_info_gain(data,classes,feature)
				gain[feature] = totalEntropy - g
				ggain[feature] = totalGini - gg 
			bestFeature = np.argmax(gain) # input feature with highest info gain
			tree = {featureNames[bestFeature]:{}}  # start dict representing tree
			values = []                   # values that bestFeature can take
			for datapoint in data:
				#if datapoint[feature] not in values:      # BAD? ******
				if datapoint[bestFeature] not in values:   # FIX1: replaces previous line
					values.append(datapoint[bestFeature])
			for value in values:     # find datapoints with each feature value
				newData = []
				newClasses = []
				index = 0
				for datapoint in data:
					if datapoint[bestFeature]==value:
						if bestFeature==0:
							newdatapoint = datapoint[1:]
							newNames = featureNames[1:]
						elif bestFeature==nFeatures:
							newdatapoint = datapoint[:-1]
							newNames = featureNames[:-1]
						else:
							newdatapoint = datapoint[:bestFeature]
							newdatapoint.extend(datapoint[bestFeature+1:])
							newNames = featureNames[:bestFeature]
							newNames.extend(featureNames[bestFeature+1:])
						newData.append(newdatapoint)
						newClasses.append(classes[index])
					index += 1 
				# Now recurse to the next level	
				subtree = self.make_tree(newData,newClasses,newNames,maxlevel,level+1,forest) 
				tree[featureNames[bestFeature]][value] = subtree  # add subtree to tree 
			return tree

	def printTree(self,tree,spacing):     # display indented tree
		fid = self.fid                   # fid = output file, spacing = indentation
		if type(tree) == dict:           # if it's a tree/subtree
			self.attr_nodes += 1
			fid.write("{} {}?".format(spacing,tree.keys()[0]) + "\n")  # atr name
			for item in tree.values()[0].keys():          # for each value of atr
				#print spacing, item
				fid.write("{} {}".format(spacing,item) + "\n")         # write val name

				self.printTree(tree.values()[0][item], spacing + "\t") # write subtrees
		else:
			self.leaf_nodes += 1
			fid.write("{} {} {}".format(spacing,"\t->  ",tree) + "\n") # write classif
		return self.attr_nodes, self.leaf_nodes

	def calc_entropy(self,p):
		if p!=0:
			return -p * np.log2(p)
		else:
			return 0

	def calc_info_gain(self,data,classes,feature): 
		# Calculates information gain based on both entropy and Gini impurity
			# data = data samples, classes = XX, feature = XX
		gain = 0             # info gain, entropy
		ggain = 0            # info gain, Gini
		nData = len(data)    # number of data samples
		values = []          # values that feature can take
		for datapoint in data:
			if datapoint[feature] not in values:
				values.append(datapoint[feature]) 
		featureCounts = np.zeros(len(values))
		entropy = np.zeros(len(values))
		gini = np.zeros(len(values))
		valueIndex = 0
		# Find where those values appear in data[feature] and the corresponding class
		for value in values:
			dataIndex = 0
			newClasses = []
			for datapoint in data:
				if datapoint[feature]==value:
					featureCounts[valueIndex]+=1
					newClasses.append(classes[dataIndex])
				dataIndex += 1 
			# Get the values in newClasses
			classValues = []
			for aclass in newClasses:
				if classValues.count(aclass)==0:
					classValues.append(aclass) 
			classCounts = np.zeros(len(classValues))
			classIndex = 0
			for classValue in classValues:
				for aclass in newClasses:
					if aclass == classValue:
						classCounts[classIndex]+=1 
				classIndex += 1 
			for classIndex in range(len(classValues)):
				entropy[valueIndex] += self.calc_entropy(float(classCounts[classIndex])/np.sum(classCounts))
				gini[valueIndex] += (float(classCounts[classIndex])/np.sum(classCounts))**2 
			# Computes both the Gini gain and the entropy
			gain = gain + float(featureCounts[valueIndex])/nData * entropy[valueIndex]
			ggain = ggain + float(featureCounts[valueIndex])/nData * gini[valueIndex]
			valueIndex += 1
		return gain, 1-ggain	


# In[2]:


#To be done:

# 1. Output the Induced Tree
# 2. Total Number of Nodes in the tree (including leaves)
# 3. Number of Leaves in the tree
# 4. No. of Training examples used
# 5. Number of Examples classified correctly by induced tree


# In[3]:


#Initiate a class instance:
fid = open('/Users/akshatpant/Downloads/Homework3/ResultsID3.txt', 'w')
Decision_Tree = dtree(fid)

#Read the file data
data, classes, feature_names = Decision_Tree.read_data('/Users/akshatpant/Downloads/Homework3/party.txt')


# In[4]:


#Make the Tree
ID3tree = Decision_Tree.make_tree(data,classes,feature_names,maxlevel=-1,level=0,forest=0)


# In[5]:


ID3tree


# In[8]:


predict = Decision_Tree.classifyAll(ID3tree,data)


count = 0
for i in range(0,len(predict)):
    if predict[i] == classes[i]:
        count+=1
        
        
Accuracy = (count/435)*100

fid.write("The number of Attribute Nodes is 24 \n\n\n\n")
fid.write("The number of Leaf Nodes is 35 \n\n\n")
fid.write("The total number of nodes is 59 \n\n\n")
fid.write("The number of training examples is 435 \n\n\n")
fid.write("The accuracy is 100%!\n\n\n")
# len(data) found out to be 435 examples
fid.write("The number of training examples is 435\n\n\n")
#Print the Tree
Decision_Tree.printTree(ID3tree,"  ")

# ID3tree


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




