import cv2
import numpy as np 
import matplotlib.pyplot as plt
import random
from functions import *

'''
Photo Memory Reduction Algorithm
Resizes image to be WxH pixels, and include only
K different colours.
'''

K = 16
W = 128
H = W
epochs = 50

image = cv2.imread('heavy.png', 1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
reduced = cv2.resize(image, (W,H), interpolation = cv2.INTER_NEAREST)

#Resize numpy array for K-means algorithm
X = np.resize(reduced, (W*H, 3))
#print(X[[1 3 4]])
#Randomly initialize K centroids
initial_centroids = (np.random.rand(K,3))*255
centroids, idx = runkMeans(X, initial_centroids, epochs)

#print(centroids)

X = updateImage(X, idx, centroids)

final = np.resize(X, (128,128,3))

Titles = ["Original", "Reduced"]
images = [image, final]
count = 2

for i in range(count):
	plt.subplot(2, 2, i+1)
	plt.title(Titles[i])
	plt.imshow(images[i])

plt.show()
