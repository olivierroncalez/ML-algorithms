# k-means.py
# roncalez-olivier 
#
# Running k-means on the iris dataset.
#
# Code draws from:
#
# http://scikit-learn.org/stable/auto_examples/tree/plot_iris.html

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from scipy.spatial import distance

# Load the data
iris = load_iris()
X = iris.data [:, [1, 3]] # 1 and 3 are the features we will use here.
y = iris.target

# Find the min and max of both features
x0_min, x0_max = X[:, 0].min(), X[:, 0].max()
x1_min, x1_max = X[:, 1].min(), X[:, 1].max()


# =============================================================================
# Plotting features
# =============================================================================

plt.subplot( 1, 2, 1 )
# Plot the original data 
plt.scatter(X[:, 0], X[:, 1], c=y.astype(np.float))
# Label axes
plt.xlabel( iris.feature_names[1], fontsize=10 )
plt.ylabel( iris.feature_names[3], fontsize=10 )

plt.show()


#==============================================================================
# K-Means Algorithm
#==============================================================================


# Finding the closest distance 
k = 3
centroids = []

# Generating random centroids
for i in range(k):
    center = [(np.random.randint(x0_min, x0_max), np.random.randint(x1_min, x1_max))]
    centroids.append(center)

# K-Means algorithm
iterate = True
counter = 0

while iterate:
    counter += 1
    # Empty list to store cluster members 
    cluster_membership = []
    # Iterate over all data
    for i in range(len(X)):
        distances = []
        # For each data point, find the closest cluster. If two clusters are identical, the 
        # algorithm will pick the first cluster
        for n in range(len(centroids)):
            # Using Euclidean distance to measure cluster closeness. First calculate the distance
            # between one point and all clusters
            dst = distance.euclidean(X[i], centroids[n])
            distances.append(dst)
        # Find the index of the min cluster distance
        indexer = distances.index(min(distances))
        # Append that instance with the cluster index
        cluster_membership.append((X[i], indexer))
     
    # Empty list to store the clusters data points
    c_1 = []
    c_2 = []
    c_3 = []
    
    # Iterating over the cluster membership list and retrieving the index. Based on the index 
    # retrieved, it will assign instances to their nearest cluster.
    for i in range(len(cluster_membership)):
        if cluster_membership[i][1] == 0:
            c_1.append(cluster_membership[i][0])
        elif cluster_membership[i][1] == 1:
            c_2.append(cluster_membership[i][0])
        else:
            c_3.append(cluster_membership[i][0])
    
    # Recomputing the cluster means by column (multivariate means)
    c1_n_mean = np.mean(c_1, axis = 0)
    c2_n_mean = np.mean(c_2, axis = 0)
    c3_n_mean = np.mean(c_3, axis = 0)

    # Saving old centroids (needed for convergence checks)
    old_centroids = list(centroids)
    
    # If the new mean is not 0 (i.e., at least one instance was assigned to that cluster), assign
    # the newly computer cluster means to that cluster
    if not np.isnan(c1_n_mean).all():
        centroids[0] = c1_n_mean
    if not np.isnan(c2_n_mean).all():
        centroids[1] = c2_n_mean
    if not np.isnan(c3_n_mean).all():
        centroids[2] = c3_n_mean
    
    # Empty list for convergence checks
    moving_distance = []
    
    # For each centroid, calculate the distance between the old center and new center
    for i in range(len(centroids)):    
         moving_distance.append(distance.euclidean(centroids[i], old_centroids[i]))
    
    # If the new center does not change more than 0.00005 for each centroid, the algorithm is 
    # complete
    print moving_distance
    if all(i <= 0.00005 for i in moving_distance):
        # End the algorithm
        iterate = False
    

# Save the labels in a list
labels = []
# Iterate over the cluster membership to retrieve assigned labels (i.e., the index in the algorithm
# above)
for i in range(len(cluster_membership)):
    labels.append(cluster_membership[i][1])

# K-Means metrics. Comute the Rand score for cluster accuracy
from sklearn import metrics
kM_imp = metrics.adjusted_rand_score(y, np.array(labels))

# Printout of accuracy
print ("Rand index for k-Means implementation: %.4f" % kM_imp)


# =============================================================================
# Scikit Learn Comparison
# =============================================================================

# Import and run k-means using scikit learn
from sklearn.cluster import KMeans
kmeans = KMeans(n_clusters=3, random_state=0).fit(X)
kmeans.cluster_centers_ # Examine cluster centers after convergence
np.array(centroids) # Comparing with scratch cluster centers. Note order may not be the same


# Computing rand score from scikit learn
kM_sklearn = metrics.adjusted_rand_score(y, kM_labels)
print ("Rand index for k-Means from scikit-learn: %.4f\n" % (kM_sklearn))


# Print out of performance
if kM_sklearn > kM_imp:
    print ("Scikit-learn k-Means has a better perfomance\n")
elif kM_sklearn < kM_imp:
    print ("Your k-Means implementation has a better perfomance\n")
else:
    print ("Scikit-learn k-Means and my implementation have the same performance\n")











