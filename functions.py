import numpy as np

#Assigns each data point to closest centroid
#Stores index in new array
def findClosestCentroids(X, centroids):

	#initialize indx array and K
	idx = np.zeros((X.shape[0], 1))
	K = centroids.shape[0]

	for i in range(X.shape[0]):
		
		#compute distance between centroids and current data point
		temp = centroids - X[i]
		dist = np.zeros((K,1))

		for j in range(K):
			dist[j] = np.dot(temp[j], temp[j])

		#assign idx[i] to be index of closest centroid
		idx[i] = np.argmin(dist)

	return idx


#Computes new centroid, which is average of all 
#values assigned to centroid
def computeCentroids(X, idx, K):

	centroids = np.empty((1,3))

	for i in range(K):
		#find index of all centroids points
		I = np.where(idx == i)
		C = np.size(I[0])

		if C == 0:
			#Skip centroid, no data was assigned
			continue

		else:
			#compute average of centroids assigned values
			#assign new centroid value
			centroids = np.append(centroids, [(np.sum(X[I[0]], axis=0)/C)], axis=0)

	#remove the extra element from initialization
	return np.delete(centroids, 0, 0)



def runkMeans(X, initial_centroids, max_iters):
	'''
	Runs K-Means algorithm on data matrix X, where each row
	of X is a single example (R,G,B). Uses initial_centroids as 
	initial centroids. max_iters specififes number of iterations
	of K-Means to run.
	'''

	#Initializations
	(m,n) = X.shape
	K = initial_centroids.shape[0]
	centroids = initial_centroids
	idx = np.zeros((m,1))

	#Run K-Means
	for i in range(max_iters):

		#print('K-Means iteration ' + i + '/' + max_iters + '\n')

		idx = findClosestCentroids(X, centroids)
		centroids = computeCentroids(X, idx, K)
		#print(centroids)

	return centroids, idx


def updateImage(X, idx, centroids):

	K = centroids.shape[0]

	for i in range(K):
		I = np.where(idx == i)
		X[I[0]] = centroids[i]

	return X