#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jun  5 15:22:50 2018

@author: Olivier
"""




#==============================================================================
# kNN Algorithm
#==============================================================================
from sklearn.model_selection import train_test_split
import operator
from scipy.spatial import distance
import scipy.stats as ss
from sklearn import datasets


# Loading data
iris = datasets.load_iris()
X = iris.data[:,:] # all features included
y = iris.target

# Randomly splitting the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)


# Creating necessary functions for kNN

## Getting distance between a test instance and training data
def getNeighbors(trainingSet, testInstance, target_train, k):
	distances = []
	for x in range(len(trainingSet)):
		# Euclidean distance
		dist = distance.euclidean(testInstance, trainingSet[x])
		# Saving the training instance, training target, and distance to test instance
		distances.append((trainingSet[x], target_train[x], dist))
	# Sort the distances after calculating all distances
	distances.sort(key=operator.itemgetter(2)) # Sort
	neighbors = []
	for x in range(k):
		# Appending the k nearest neighbors
		neighbors.append(distances[x][0:2])
	# Returning tuple of nearest neighbors
	return neighbors


# Function for determining majority class label based on nearest neighbors
# Used for making classification decision.
def Categorization(neighbors):
    category = []
	# For each neighbor append their decision to empty list
    for x in range(len(neighbors)):
        category.append(neighbors[x][1])
	# Retrieve the mode of the decisions
    predictions = ss.mode(category)
    return predictions[0][0] # Return this decision

## Metrics
def getAccuracy(y_test, predictions):
	correct = 0
	# For each test target value, evaluate concordance and score +1 for positive matches
	for x in range(len(y_test)):
		if y_test[x] == predictions[x]:
			correct += 1
	# Return the number of correct answers divided by the length of the test set
	return (correct/float(len(y_test))) * 100.0

## Compilation of function
def kNN_Scratch(trainingSet, testSet, target_train, target_test, k):
    predictions = []
    for x in range(len(testSet)):
        testInstance = testSet[x]
        neighbors = getNeighbors(trainingSet, testInstance, target_train, k) # Get k nearest n's
        result = Categorization(neighbors) # Compute the classification
        predictions.append(result) # Create a list of classification decisions
    accuracy = getAccuracy(target_test, predictions)
    print('Accuracy: ' + repr(accuracy) + '%')


# Testing kNN on Iris data
kNN_Scratch(X_train, X_test, y_train, y_test, 5)



#==============================================================================
# sklearn kNN Algorithm
#==============================================================================

# Comparison with scikit learn
from sklearn.neighbors import KNeighborsClassifier

# Create classifier class
knn = KNeighborsClassifier(n_neighbors=5)
# Identify the training data (there is no classifier 'training' for kNN)
knn.fit(X_train, y_train)
# Score the test set
score = knn.score(X_test, y_test)
print "Accuracy: %s" %(score)
# Confirmation of same accuracy score
